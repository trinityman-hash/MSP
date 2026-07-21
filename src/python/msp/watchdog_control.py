"""
Give a Python caller the same telemetry-driven rollback guarantee
sandbox_watchdog.c gives a monitored C thread -- this is docs/STATUS.md's
"what's left to do" item #1: "give Python a way to participate in
sandbox_watchdog's own fallback/control signal, not just its telemetry."

WHY THIS ISN'T A DIRECT BINDING OF MSP_ARM_FALLBACK_POINT()
-------------------------------------------------------------
The C macro (src/daemon/sandbox_watchdog.h) works by textually inlining
sigsetjmp() into the monitored thread's own stack frame, so a later
siglongjmp() (run from the watchdog's SIGUSR1 handler, in the interrupted
thread's own context) has a live, valid frame to unwind to. That header's
own comment documents a second, subtler bug that was found and fixed
while building it: wrapping sigsetjmp() in an ordinary helper function is
*already* broken, because the helper's frame is popped on its normal
return, long before the signal arrives.

A Python callable can't be armed the same way. Calling into a live
CPython interpreter loop and later asynchronously unwinding through it
from a signal handler is a fundamentally different, and fundamentally
less safe, proposition than unwinding a bounded, self-contained C stack:
siglongjmp() has no notion of reference counts, the GIL, or any
Python/C-level cleanup that should run for objects on the frames it
discards. Jumping out of arbitrary, in-flight interpreter execution risks
corrupting the interpreter's own state, not just abandoning the training
step -- a strictly worse outcome than the thermal/security violation the
mechanism exists to protect against. (This is also why CPython's own
signal handling defers actual Python-level work to a flag the interpreter
checks between bytecode instructions, rather than acting synchronously
inside the handler.)

This module instead implements the other option docs/STATUS.md names for
this problem: a process boundary. The protected work runs in a child
process; a violation is handled by terminating that whole process
(SIGTERM, escalating to SIGKILL after a grace period) rather than by
unwinding a live stack in place. A killed process is always safe for the
OS to reclaim -- there is no analogue of "corrupted interpreter state"
left behind in the parent, which never touches the risky work directly.
The trade-off, versus the C mechanism, is coarser granularity (the whole
child process is lost, not just one in-flight computation) and the
ordinary cost of a process boundary (the protected callable must be
picklable, and its result crosses back via a Queue) -- both explicit,
honest trade-offs rather than a fragile reproduction of the C approach.

Telemetry can be shared with the real watchdog daemon via
`telemetry_probe_from_watchdog_socket` (built on
msp.thermal.WatchdogTelemetryReader), so the same violation that would
trigger sandbox_watchdog.c's own fallback is what triggers this one --
not a second, independent policy that could disagree with it.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import queue as _queue_module
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple

from .thermal import WatchdogTelemetryReader

# Mirrors sandbox_watchdog.h's MSP_CRITICAL_TEMP_C. Not imported directly
# -- there is no shared header between the C and Python layers -- so this
# is a separately-maintained constant, same as DEFAULT_MEMORY_BUDGET_BYTES
# in adapter_manager.py mirrors the C allocator's own 512MB budget.
DEFAULT_CRITICAL_TEMP_C = 85.0

DEFAULT_WATCHDOGD_SOCKET_PATH = "/tmp/msp_watchdogd.sock"


class FallbackReason(Enum):
    """Why a WatchdogGuardedExecutor.run() call terminated its child early."""

    TEMPERATURE = "temperature"
    SRAM = "sram"
    SECURITY = "security"
    TELEMETRY_ERROR = "telemetry_error"


@dataclass(frozen=True)
class TelemetryReading:
    """One point-in-time sample, structurally matching the C daemon's
    msp_telemetry_t (temperature_c, sram_usage_bytes, security_violation)
    plus a monotonic timestamp for observability."""

    temperature_c: float
    sram_usage_bytes: int
    security_violation: bool
    timestamp: float


TelemetryProbe = Callable[[], TelemetryReading]


def telemetry_probe_from_watchdog_socket(
    socket_path: str = DEFAULT_WATCHDOGD_SOCKET_PATH, timeout_s: float = 2.0
) -> TelemetryProbe:
    """
    Builds a TelemetryProbe backed by a running `watchdogd`'s telemetry
    socket (see src/daemon/watchdogd_main.c), via
    msp.thermal.WatchdogTelemetryReader -- the same real telemetry source
    sandbox_watchdog.c itself acts on, not a second independent read.
    Prefer this over `telemetry_probe_from_reader` whenever `watchdogd` is
    available, so this executor's fallback decisions track the native
    watchdog's exactly.

    Caveat, discovered while testing this module: if the real telemetry
    stays above `watchdogd`'s own hardcoded MSP_CRITICAL_TEMP_C (85.0C,
    see sandbox_watchdog.h) for a sustained period, `watchdogd`'s own
    watchdog thread repeatedly detects that violation and cycles
    `telemetry_server` through its own fallback-and-cooldown sequence
    (see watchdogd_main.c), which makes the socket intermittently refuse
    or time out new connections for the duration -- precisely while the
    condition a caller most needs telemetry for is ongoing. A
    ViolationPolicy configured to react at or below that same 85C will
    usually see a reading (or a connection failure, handled per
    `fail_safe_on_telemetry_error`) before the cycle makes the socket
    unavailable, but this is a real characteristic of the shared daemon,
    not a bug in this probe -- worth knowing if a policy's threshold is
    set very close to 85C.
    """
    reader = WatchdogTelemetryReader(socket_path=socket_path, timeout_s=timeout_s)

    def _probe() -> TelemetryReading:
        temperature_c, sram_usage_bytes, security_violation = reader.read_full()
        return TelemetryReading(temperature_c, sram_usage_bytes, security_violation, time.monotonic())

    return _probe


def telemetry_probe_from_reader(reader: Callable[[], float]) -> TelemetryProbe:
    """
    Adapts a plain, temperature-only reader (e.g. a ScriptedThermalReader
    or LinuxThermalZoneReader from msp.thermal -- anything usable as
    ThermalMonitor(reader=...)) into a full TelemetryProbe. sram_usage_bytes
    is always reported as 0 and security_violation as always False, since a
    temperature-only reader has no way to observe either; use
    `telemetry_probe_from_watchdog_socket` if those matter.
    """

    def _probe() -> TelemetryReading:
        return TelemetryReading(reader(), 0, False, time.monotonic())

    return _probe


@dataclass(frozen=True)
class ViolationPolicy:
    """
    Decides whether a TelemetryReading counts as a violation that should
    terminate the protected child process.

    Parameters
    ----------
    temperature_threshold_c:
        Defaults to DEFAULT_CRITICAL_TEMP_C (85.0), matching
        sandbox_watchdog.h's MSP_CRITICAL_TEMP_C.
    sram_usage_threshold_bytes:
        If given, a reading reporting more than this many bytes counts as
        a violation. None (default) disables this check -- most callers
        using telemetry_probe_from_reader will never populate
        sram_usage_bytes anyway (see its docstring), and Python-side
        memory budgeting is already AdapterManager's job.
    trigger_on_security_violation:
        If True (default), any reading with security_violation=True is an
        immediate violation regardless of temperature/SRAM.
    poll_interval_s:
        How often the executor's run loop samples telemetry while the
        child is alive.
    fail_safe_on_telemetry_error:
        If True (default), an exception raised by the TelemetryProbe
        itself (e.g. WatchdogTelemetryReader losing its socket) is treated
        as a violation (FallbackReason.TELEMETRY_ERROR) rather than
        silently skipped. This mirrors thermal_reader.c's own documented
        choice to report +INFINITY on a failed hardware read: a broken
        telemetry source should be able to cause an unnecessary fallback,
        never silently mask a real one by being treated as "no news, so
        assume it's fine." Set False only if occasional probe failures
        are expected and should not interrupt the protected work.
    """

    temperature_threshold_c: float = DEFAULT_CRITICAL_TEMP_C
    sram_usage_threshold_bytes: Optional[int] = None
    trigger_on_security_violation: bool = True
    poll_interval_s: float = 0.5
    fail_safe_on_telemetry_error: bool = True

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if self.sram_usage_threshold_bytes is not None and self.sram_usage_threshold_bytes < 0:
            raise ValueError("sram_usage_threshold_bytes must be non-negative")

    def evaluate(self, reading: TelemetryReading) -> Optional[FallbackReason]:
        """Returns the violated FallbackReason, or None if `reading` is
        within policy. Checked in order of severity: an active security
        violation always wins over a temperature or SRAM reading, even if
        those happen to also be within bounds."""
        if self.trigger_on_security_violation and reading.security_violation:
            return FallbackReason.SECURITY
        if reading.temperature_c > self.temperature_threshold_c:
            return FallbackReason.TEMPERATURE
        if (
            self.sram_usage_threshold_bytes is not None
            and reading.sram_usage_bytes > self.sram_usage_threshold_bytes
        ):
            return FallbackReason.SRAM
        return None


class GuardedProcessError(RuntimeError):
    """
    Raised on the parent side (via ProtectedRunResult.exception /
    raise_if_error()) when the child process's target callable raised.

    Deliberately wraps rather than re-raises the original exception
    object: the original type may not exist, or may not be reconstructible
    from a plain message, in the parent's process (it crossed a pickling
    boundary), so faithfully reproducing "raise SomeLibrarySpecificError"
    here would be unreliable in general. `original_type_name` and
    `child_traceback` preserve the diagnostic information instead.
    """

    def __init__(self, original_type_name: str, message: str, child_traceback: str) -> None:
        super().__init__(f"{original_type_name}: {message}")
        self.original_type_name = original_type_name
        self.child_traceback = child_traceback


@dataclass
class ProtectedRunResult:
    """Outcome of a WatchdogGuardedExecutor.run() call."""

    completed: bool
    fallback_triggered: bool
    fallback_reason: Optional[FallbackReason] = None
    violation_telemetry: Optional[TelemetryReading] = None
    return_value: Any = None
    exception: Optional[BaseException] = None
    """Set in two cases: (1) the child's target callable raised (wrapped
    as GuardedProcessError), or (2) a fallback was triggered by
    FallbackReason.TELEMETRY_ERROR, in which case this carries the
    telemetry probe's own exception message for diagnosis. Case (2) is
    the only situation where `fallback_triggered` and `exception` are
    both set at once."""
    elapsed_s: float = 0.0

    def raise_if_error(self) -> None:
        """Re-raises `exception` if the child completed but its target
        callable raised. A no-op otherwise -- including when
        fallback_triggered is True, since a triggered fallback is a
        successful safety response, not an error; check
        `fallback_triggered` for that separately."""
        if self.exception is not None:
            raise self.exception


def _child_entrypoint(
    target: Callable[..., Any], args: Sequence[Any], kwargs: dict, result_queue: "mp.Queue"
) -> None:
    """
    Runs in the child process. Must be a module-level function (not a
    closure/lambda/method) so it is picklable under the "spawn" start
    method -- see WatchdogGuardedExecutor.run's docstring for why spawn,
    not fork, is used.
    """
    try:
        value = target(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentionally broad; reported, not swallowed
        result_queue.put(("error", type(exc).__name__, str(exc), traceback.format_exc()))
        return

    # multiprocessing.Queue.put() only *enqueues* the object; the actual
    # pickling happens later, in the Queue's internal background feeder
    # thread. A pickling failure there is logged to stderr by that thread
    # and otherwise silently dropped -- it would never reach a try/except
    # wrapped around put() itself. Pickling explicitly and synchronously,
    # here, is what makes an unpicklable return value a reported
    # GuardedProcessError on the parent side instead of an unexplained
    # "child exited without producing a result."
    try:
        payload = pickle.dumps(value)
    except Exception as exc:
        result_queue.put(
            ("error", "PicklingError", f"return value could not be pickled: {exc}", traceback.format_exc())
        )
        return
    result_queue.put(("ok", payload))


class WatchdogGuardedExecutor:
    """
    Runs a callable in a child process, terminating it early if
    `telemetry_source` reports a violation per `policy` -- see the module
    docstring for the full rationale versus a direct C-mechanism binding.

    Example
    -------
    >>> from msp.watchdog_control import (
    ...     WatchdogGuardedExecutor, ViolationPolicy, telemetry_probe_from_watchdog_socket,
    ... )
    >>> executor = WatchdogGuardedExecutor(
    ...     telemetry_source=telemetry_probe_from_watchdog_socket(),
    ...     policy=ViolationPolicy(temperature_threshold_c=80.0),
    ... )
    >>> result = executor.run(my_training_step, args=(adapter_id,))
    >>> if result.fallback_triggered:
    ...     roll_back_to_base_model()
    ... else:
    ...     result.raise_if_error()  # re-raises if my_training_step raised
    ...     use(result.return_value)
    """

    def __init__(self, telemetry_source: TelemetryProbe, policy: Optional[ViolationPolicy] = None) -> None:
        self._telemetry_source = telemetry_source
        self._policy = policy or ViolationPolicy()

    def run(
        self,
        target: Callable[..., Any],
        args: Sequence[Any] = (),
        kwargs: Optional[dict] = None,
        *,
        grace_period_s: float = 2.0,
    ) -> ProtectedRunResult:
        """
        Runs `target(*args, **kwargs)` in a child process, polling
        `telemetry_source` roughly every `policy.poll_interval_s` while it
        runs.

        `target` (and everything in `args`/`kwargs`) must be picklable --
        this uses multiprocessing's "spawn" start method explicitly
        (rather than the platform default, which is "fork" on Linux) so
        the child gets a fresh interpreter instead of a copy-on-write
        snapshot of the parent's, including whatever threads, open file
        descriptors, and (notably) CUDA/PyTorch state the parent process
        may hold. Forking a multi-threaded process that holds a CUDA
        context is a well-known source of hangs and corruption; spawn
        avoids that class of bug entirely, at the cost of the child
        re-importing modules on startup.

        If a violation is detected, the child receives SIGTERM
        immediately; if it hasn't exited within `grace_period_s`, it is
        escalated to SIGKILL. Either way this method always returns (it
        does not raise for a violation) -- check the returned
        ProtectedRunResult.
        """
        kwargs = kwargs or {}
        ctx = mp.get_context("spawn")
        result_queue: "mp.Queue" = ctx.Queue()
        child = ctx.Process(target=_child_entrypoint, args=(target, args, kwargs, result_queue), daemon=False)
        start = time.monotonic()
        child.start()

        try:
            fallback_reason: Optional[FallbackReason] = None
            violation_telemetry: Optional[TelemetryReading] = None
            telemetry_error: Optional[BaseException] = None
            child_result: Optional[Tuple] = None

            while True:
                try:
                    child_result = result_queue.get(timeout=self._policy.poll_interval_s)
                    break  # child finished (ok or error) before any violation
                except _queue_module.Empty:
                    pass

                if not child.is_alive():
                    # Exited without publishing a result: a crash, or an
                    # external kill. One last non-blocking drain in case
                    # of a race between is_alive() going False and the
                    # queue write becoming visible.
                    try:
                        child_result = result_queue.get_nowait()
                    except _queue_module.Empty:
                        child_result = None
                    break

                try:
                    reading = self._telemetry_source()
                except Exception as exc:  # noqa: BLE001 - policy decides fail-open vs fail-closed
                    if not self._policy.fail_safe_on_telemetry_error:
                        continue
                    fallback_reason = FallbackReason.TELEMETRY_ERROR
                    violation_telemetry = TelemetryReading(
                        temperature_c=float("inf"),
                        sram_usage_bytes=0,
                        security_violation=False,
                        timestamp=time.monotonic(),
                    )
                    telemetry_error = exc
                    break

                reason = self._policy.evaluate(reading)
                if reason is not None:
                    fallback_reason = reason
                    violation_telemetry = reading
                    break

            if fallback_reason is not None:
                self._terminate(child, grace_period_s)
                return ProtectedRunResult(
                    completed=False,
                    fallback_triggered=True,
                    fallback_reason=fallback_reason,
                    violation_telemetry=violation_telemetry,
                    elapsed_s=time.monotonic() - start,
                    exception=(
                        RuntimeError(f"telemetry probe failed: {telemetry_error}")
                        if telemetry_error is not None
                        else None
                    ),
                )

            if child_result is None:
                # No violation, and the fast path above didn't get a
                # result yet (child likely finished between a queue
                # timeout and the is_alive() check). Give it a bounded
                # window to finish publishing.
                try:
                    child_result = result_queue.get(timeout=grace_period_s)
                except _queue_module.Empty:
                    child_result = None

            self._terminate(child, grace_period_s)  # no-op if already exited cleanly
            elapsed = time.monotonic() - start

            if child_result is None:
                return ProtectedRunResult(
                    completed=False,
                    fallback_triggered=False,
                    elapsed_s=elapsed,
                    exception=RuntimeError(
                        f"guarded process exited without producing a result "
                        f"(exit code {child.exitcode})"
                    ),
                )

            status = child_result[0]
            if status == "ok":
                return ProtectedRunResult(
                    completed=True,
                    fallback_triggered=False,
                    return_value=pickle.loads(child_result[1]),
                    elapsed_s=elapsed,
                )
            _, exc_type_name, exc_message, exc_traceback = child_result
            return ProtectedRunResult(
                completed=True,
                fallback_triggered=False,
                elapsed_s=elapsed,
                exception=GuardedProcessError(exc_type_name, exc_message, exc_traceback),
            )
        finally:
            # Belt-and-suspenders: guarantee no orphaned child process
            # survives this call, regardless of which path was taken above
            # (including one raising unexpectedly, e.g. a broken
            # telemetry_source with fail_safe_on_telemetry_error=False
            # masking every error rather than a single expected exc type).
            if child.is_alive():
                child.kill()
                child.join(timeout=grace_period_s)
            result_queue.close()
            result_queue.join_thread()

    @staticmethod
    def _terminate(child: "mp.process.BaseProcess", grace_period_s: float) -> None:
        """SIGTERM, then SIGKILL after grace_period_s if still alive. This
        is the Python-side analogue of sandbox_watchdog.c signaling
        MSP_FALLBACK_SIGNAL to the monitored thread -- but unlike that
        signal (SIGUSR1, caught by a cooperative sigsetjmp handler), a
        plain terminating signal is appropriate here: the child is not
        expected to install a matching handler and roll back in place, it
        is simply reclaimed."""
        if not child.is_alive():
            return
        child.terminate()  # SIGTERM
        child.join(timeout=grace_period_s)
        if child.is_alive():
            child.kill()  # SIGKILL
            child.join(timeout=grace_period_s)

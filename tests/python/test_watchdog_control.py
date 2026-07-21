"""
Tests for msp.watchdog_control -- see that module's docstring for why
this exists (docs/STATUS.md item #1) and why it's a process boundary
rather than a direct binding of sandbox_watchdog.c's
MSP_ARM_FALLBACK_POINT()/MSP_FALLBACK_SIGNAL.

Child-process targets used by multiprocessing's "spawn" start method must
be importable module-level functions (not closures/lambdas), so the
targets used across these tests are defined here at module scope rather
than inline in each test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

import pytest

from msp.watchdog_control import (
    DEFAULT_CRITICAL_TEMP_C,
    FallbackReason,
    GuardedProcessError,
    TelemetryReading,
    ViolationPolicy,
    WatchdogGuardedExecutor,
    telemetry_probe_from_reader,
    telemetry_probe_from_watchdog_socket,
)


# --- module-level child targets (must be picklable under spawn) ---


def _add(a, b):
    return a + b


def _raise_value_error(message):
    raise ValueError(message)


def _return_unpicklable():
    # A local (non-module-level) class instance can't be pickled by the
    # default pickler -- this is what should surface as a PicklingError
    # result rather than a silently lost value.
    class _NotPicklable:
        pass

    return _NotPicklable()


def _heartbeat_loop(path, iterations=200, interval_s=0.02):
    """Writes an increasing counter to `path` every `interval_s` --
    used to prove a child was actually terminated (the file stops
    advancing) rather than merely having its result ignored."""
    for i in range(iterations):
        with open(path, "w") as f:
            f.write(str(i))
        time.sleep(interval_s)
    return "finished-without-being-killed"


def _ignore_sigterm_and_loop(path, iterations=200, interval_s=0.02):
    """Like _heartbeat_loop, but ignores SIGTERM first -- used to prove
    the executor escalates to SIGKILL rather than hanging forever."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for i in range(iterations):
        with open(path, "w") as f:
            f.write(str(i))
        time.sleep(interval_s)
    return "finished-without-being-killed"


# --- a stateful-but-picklable-source helper (kept in the PARENT process
# only -- telemetry_source callables never cross into the child, so they
# may freely be closures/local state, unlike the `target` above) ---


class _ScriptedTelemetry:
    """Returns a fixed sequence of TelemetryReadings, holding the last one
    once exhausted -- the TelemetryReading analogue of
    msp.thermal.ScriptedThermalReader."""

    def __init__(self, readings):
        assert readings
        self._readings = list(readings)
        self._idx = 0

    def __call__(self):
        reading = self._readings[min(self._idx, len(self._readings) - 1)]
        self._idx += 1
        return reading


def _safe_reading():
    return TelemetryReading(temperature_c=25.0, sram_usage_bytes=0, security_violation=False, timestamp=0.0)


def _hot_reading():
    return TelemetryReading(temperature_c=95.0, sram_usage_bytes=0, security_violation=False, timestamp=0.0)


def _security_violation_reading():
    return TelemetryReading(temperature_c=25.0, sram_usage_bytes=0, security_violation=True, timestamp=0.0)


class _HotOnceHeartbeatAdvances:
    """
    Returns a safe reading until `path` shows the child has written at
    least `min_value`, then hot readings after that.

    Deliberately NOT a fixed-length scripted sequence keyed to poll
    count: multiprocessing's "spawn" start method re-imports the whole
    interpreter (including torch) in the child before it runs a single
    line of the target, and that startup latency is both real and
    environment-dependent. Tying the "become hot" transition to *observed
    child progress* instead of an assumed number of polls keeps these
    tests correct regardless of how long that startup takes.
    """

    def __init__(self, path, min_value=2):
        self._path = path
        self._min_value = min_value

    def __call__(self):
        try:
            content = open(self._path).read().strip()
            value = int(content) if content else -1
        except (FileNotFoundError, ValueError, OSError):
            value = -1
        return _hot_reading() if value >= self._min_value else _safe_reading()


# --- ViolationPolicy ---


def test_policy_defaults_match_c_critical_temp():
    assert ViolationPolicy().temperature_threshold_c == DEFAULT_CRITICAL_TEMP_C


def test_policy_rejects_nonpositive_poll_interval():
    with pytest.raises(ValueError):
        ViolationPolicy(poll_interval_s=0.0)


def test_policy_rejects_negative_sram_threshold():
    with pytest.raises(ValueError):
        ViolationPolicy(sram_usage_threshold_bytes=-1)


def test_policy_evaluate_prioritizes_security_over_temperature():
    policy = ViolationPolicy(temperature_threshold_c=1000.0)  # temp check would never fire
    reading = TelemetryReading(temperature_c=25.0, sram_usage_bytes=0, security_violation=True, timestamp=0.0)
    assert policy.evaluate(reading) is FallbackReason.SECURITY


def test_policy_evaluate_returns_none_when_within_bounds():
    assert ViolationPolicy().evaluate(_safe_reading()) is None


def test_policy_security_check_can_be_disabled():
    policy = ViolationPolicy(trigger_on_security_violation=False)
    assert policy.evaluate(_security_violation_reading()) is None


# --- telemetry_probe_from_reader ---


def test_probe_from_reader_reports_zero_sram_and_no_security_violation():
    probe = telemetry_probe_from_reader(lambda: 42.0)
    reading = probe()
    assert reading.temperature_c == 42.0
    assert reading.sram_usage_bytes == 0
    assert reading.security_violation is False


# --- WatchdogGuardedExecutor: normal completion ---


def test_completes_normally_and_returns_value():
    executor = WatchdogGuardedExecutor(telemetry_source=_ScriptedTelemetry([_safe_reading()]))
    result = executor.run(_add, args=(2, 3))
    assert result.completed is True
    assert result.fallback_triggered is False
    assert result.return_value == 5
    assert result.exception is None


def test_child_exception_is_wrapped_and_reraised_on_demand():
    executor = WatchdogGuardedExecutor(telemetry_source=_ScriptedTelemetry([_safe_reading()]))
    result = executor.run(_raise_value_error, args=("boom",))
    assert result.completed is True
    assert result.fallback_triggered is False
    assert isinstance(result.exception, GuardedProcessError)
    assert result.exception.original_type_name == "ValueError"
    assert "boom" in str(result.exception)

    with pytest.raises(GuardedProcessError):
        result.raise_if_error()


def test_unpicklable_return_value_reported_not_silently_lost():
    executor = WatchdogGuardedExecutor(telemetry_source=_ScriptedTelemetry([_safe_reading()]))
    result = executor.run(_return_unpicklable)
    assert result.completed is True
    assert isinstance(result.exception, GuardedProcessError)
    assert result.exception.original_type_name == "PicklingError"


def test_raise_if_error_is_noop_when_no_exception():
    executor = WatchdogGuardedExecutor(telemetry_source=_ScriptedTelemetry([_safe_reading()]))
    result = executor.run(_add, args=(1, 1))
    result.raise_if_error()  # must not raise


# --- WatchdogGuardedExecutor: violation / fallback path ---


def test_temperature_violation_terminates_child_promptly(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    policy = ViolationPolicy(poll_interval_s=0.05)
    telemetry = _HotOnceHeartbeatAdvances(str(heartbeat_path), min_value=2)
    executor = WatchdogGuardedExecutor(telemetry_source=telemetry, policy=policy)

    result = executor.run(_heartbeat_loop, args=(str(heartbeat_path),), grace_period_s=1.0)

    assert result.completed is False
    assert result.fallback_triggered is True
    assert result.fallback_reason is FallbackReason.TEMPERATURE
    assert result.violation_telemetry.temperature_c == 95.0
    # The child was scheduled to run for iterations*interval_s ~= 4s;
    # a prompt kill should return well before that.
    assert result.elapsed_s < 4.0

    # Confirm the child process was actually stopped, not just ignored:
    # the heartbeat file should stop advancing.
    time.sleep(0.3)
    first = heartbeat_path.read_text()
    time.sleep(0.3)
    second = heartbeat_path.read_text()
    assert first == second


def test_security_violation_triggers_fallback_even_when_cool(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    policy = ViolationPolicy(poll_interval_s=0.05)
    telemetry = _ScriptedTelemetry([_security_violation_reading()])
    executor = WatchdogGuardedExecutor(telemetry_source=telemetry, policy=policy)

    result = executor.run(_heartbeat_loop, args=(str(heartbeat_path),), grace_period_s=1.0)

    assert result.fallback_triggered is True
    assert result.fallback_reason is FallbackReason.SECURITY


def test_sram_threshold_triggers_fallback(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    policy = ViolationPolicy(poll_interval_s=0.05, sram_usage_threshold_bytes=1024)
    over_budget = TelemetryReading(
        temperature_c=25.0, sram_usage_bytes=2048, security_violation=False, timestamp=0.0
    )
    executor = WatchdogGuardedExecutor(telemetry_source=_ScriptedTelemetry([over_budget]), policy=policy)

    result = executor.run(_heartbeat_loop, args=(str(heartbeat_path),), grace_period_s=1.0)

    assert result.fallback_triggered is True
    assert result.fallback_reason is FallbackReason.SRAM


def test_sigterm_is_escalated_to_sigkill(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    policy = ViolationPolicy(poll_interval_s=0.05)
    telemetry = _HotOnceHeartbeatAdvances(str(heartbeat_path), min_value=2)
    executor = WatchdogGuardedExecutor(telemetry_source=telemetry, policy=policy)

    result = executor.run(
        _ignore_sigterm_and_loop, args=(str(heartbeat_path),), grace_period_s=0.5
    )

    assert result.fallback_triggered is True
    # grace_period_s (0.5) for SIGTERM to work, plus another grace_period_s
    # budget for the SIGKILL escalation -- generous bound to avoid flakes.
    assert result.elapsed_s < 5.0

    time.sleep(0.3)
    first = heartbeat_path.read_text()
    time.sleep(0.3)
    second = heartbeat_path.read_text()
    assert first == second


def test_telemetry_error_defaults_to_fail_safe(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"

    def _broken_probe():
        raise RuntimeError("simulated telemetry source failure")

    policy = ViolationPolicy(poll_interval_s=0.05)
    executor = WatchdogGuardedExecutor(telemetry_source=_broken_probe, policy=policy)

    result = executor.run(_heartbeat_loop, args=(str(heartbeat_path),), grace_period_s=1.0)

    assert result.fallback_triggered is True
    assert result.fallback_reason is FallbackReason.TELEMETRY_ERROR
    assert result.violation_telemetry.temperature_c == float("inf")
    assert result.exception is not None
    assert "simulated telemetry source failure" in str(result.exception)


def test_telemetry_error_can_be_configured_to_not_fail_safe():
    calls = {"n": 0}

    def _flaky_then_safe():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _safe_reading()

    policy = ViolationPolicy(poll_interval_s=0.02, fail_safe_on_telemetry_error=False)
    executor = WatchdogGuardedExecutor(telemetry_source=_flaky_then_safe, policy=policy)

    result = executor.run(_add, args=(10, 20))

    assert result.fallback_triggered is False
    assert result.completed is True
    assert result.return_value == 30


def test_no_orphaned_process_left_after_normal_completion():
    executor = WatchdogGuardedExecutor(telemetry_source=_ScriptedTelemetry([_safe_reading()]))
    executor.run(_add, args=(1, 2))
    # multiprocessing.active_children() reaps/reports any still-tracked
    # child of this process; there should be none left over.
    import multiprocessing

    assert multiprocessing.active_children() == []


# --- integration with the real watchdogd binary / telemetry socket ---


def _find_watchdogd():
    override = os.environ.get("MSP_WATCHDOGD_BIN")
    if override and os.path.isfile(override):
        return override
    for candidate in ("build/watchdogd", "build-asan/watchdogd"):
        if os.path.isfile(candidate):
            return candidate
    return None


_WATCHDOGD_BIN = _find_watchdogd()
_SKIP_REASON = (
    "watchdogd binary not built -- run `cmake -B build && cmake --build build` "
    "first (see CMakeLists.txt's watchdogd target), or set MSP_WATCHDOGD_BIN"
)
requires_watchdogd = pytest.mark.skipif(_WATCHDOGD_BIN is None, reason=_SKIP_REASON)


def _wait_for_socket(path, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.02)
    return False


def _make_fake_zone(base_dir, index, zone_type, millidegrees):
    zone_dir = base_dir / f"thermal_zone{index}"
    zone_dir.mkdir(parents=True)
    (zone_dir / "type").write_text(zone_type)
    (zone_dir / "temp").write_text(str(millidegrees))
    return zone_dir


@requires_watchdogd
def test_watchdogd_integration_hot_zone_triggers_fallback(tmp_path):
    """End-to-end: a real (simulated-sysfs) watchdogd process is the
    telemetry source, exactly as sandbox_watchdog.c itself would read it,
    and a hot zone drives WatchdogGuardedExecutor to terminate the child
    -- not a second, independent policy that merely happens to agree.

    Uses a policy threshold (70C) below watchdogd's own hardcoded
    MSP_CRITICAL_TEMP_C (85C, see sandbox_watchdog.h) and a fake zone
    reading (75C) between the two. This is deliberate, not incidental: if
    the fake zone were set above 85C instead, watchdogd's OWN watchdog
    thread would repeatedly detect its own critical violation and cycle
    telemetry_server through fallback/cooldown (see
    watchdogd_main.c) -- observed directly while writing this test as
    repeated 'fallback triggered -- telemetry server interrupted' lines
    on watchdogd's stderr, making its socket intermittently unservable
    exactly when a client tries to read it. Staying under 85C keeps
    watchdogd's own telemetry service stable while still exercising a
    real violation against THIS executor's (lower, more conservative)
    policy -- a legitimate defense-in-depth shape: a Python-side policy
    can choose to fall back earlier than the hardware-level daemon's own
    fixed threshold, which remains a second, independent safety net
    underneath it.
    """
    thermal_dir = tmp_path / "thermal"
    _make_fake_zone(thermal_dir, 0, "cpu-thermal", 75000)  # 75.0C
    socket_path = tmp_path / "watchdogd.sock"
    heartbeat_path = tmp_path / "heartbeat.txt"

    proc = subprocess.Popen(
        [_WATCHDOGD_BIN, "--socket", str(socket_path), "--thermal-base", str(thermal_dir)],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for_socket(str(socket_path)), (
            f"watchdogd never created its socket; stderr so far:\n"
            f"{proc.stderr.read() if proc.poll() is not None else '(still running)'}"
        )

        executor = WatchdogGuardedExecutor(
            telemetry_source=telemetry_probe_from_watchdog_socket(socket_path=str(socket_path)),
            policy=ViolationPolicy(poll_interval_s=0.1, temperature_threshold_c=70.0),
        )
        result = executor.run(_heartbeat_loop, args=(str(heartbeat_path),), grace_period_s=1.0)

        assert result.fallback_triggered is True
        assert result.fallback_reason is FallbackReason.TEMPERATURE
        assert result.violation_telemetry.temperature_c == pytest.approx(75.0, abs=1e-2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)

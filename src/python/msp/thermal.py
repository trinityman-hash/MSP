"""
Thermal telemetry for training-time gradient throttling.

The original v3.0 spec hardcoded a fixed temperature (55.0) directly in the
watchdog loop, which means the "10ms telemetry constraint" could never
actually detect a real thermal event -- it was reading a constant. This
module replaces that with a pluggable reader so the same gradient-freezing
logic can be driven by real hardware telemetry, a test double, or a
deterministic simulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class ThermalReading:
    """A single point-in-time telemetry sample."""

    temperature_c: float
    timestamp: float


class ThermalMonitor:
    """
    Pluggable temperature source.

    Parameters
    ----------
    reader:
        A zero-argument callable returning the current temperature in
        Celsius. Defaults to a safe constant reader (25.0C) so unit tests
        are deterministic unless a scenario explicitly overrides it.
    freeze_threshold_c:
        Temperature above which gradient updates should be throttled.
    freeze_ratio:
        Fraction of gradient rows that remain *active* (unfrozen) once the
        threshold is exceeded. NOTE: the original spec used the term
        "update_ratio" ambiguously -- here it is explicit: 0.2 means 20%
        of rows still update (i.e. an 80% freeze), matching the spec's
        stated intent.
    """

    def __init__(
        self,
        reader: Optional[Callable[[], float]] = None,
        freeze_threshold_c: float = 75.0,
        freeze_ratio: float = 0.2,
    ) -> None:
        if not (0.0 < freeze_ratio <= 1.0):
            raise ValueError("freeze_ratio must be in (0.0, 1.0]")
        self._reader = reader or (lambda: 25.0)
        self.freeze_threshold_c = freeze_threshold_c
        self.freeze_ratio = freeze_ratio

    def read(self) -> ThermalReading:
        return ThermalReading(temperature_c=self._reader(), timestamp=time.monotonic())

    def update_ratio(self, reading: Optional[ThermalReading] = None) -> float:
        """
        Fraction of gradient rows that should receive an update this step.
        1.0 = full update, self.freeze_ratio = throttled.
        """
        reading = reading or self.read()
        return self.freeze_ratio if reading.temperature_c > self.freeze_threshold_c else 1.0

    def is_throttling(self, reading: Optional[ThermalReading] = None) -> bool:
        reading = reading or self.read()
        return reading.temperature_c > self.freeze_threshold_c


class ScriptedThermalReader:
    """
    Test double: replays a fixed sequence of temperatures, holding the last
    value once the sequence is exhausted. Useful for deterministically
    testing the >75C throttling path without real hardware.
    """

    def __init__(self, sequence: list[float]) -> None:
        if not sequence:
            raise ValueError("sequence must be non-empty")
        self._sequence = list(sequence)
        self._idx = 0

    def __call__(self) -> float:
        value = self._sequence[min(self._idx, len(self._sequence) - 1)]
        self._idx += 1
        return value


@dataclass(frozen=True)
class ThermalZoneInfo:
    """One discovered Linux thermal zone."""

    index: int
    zone_type: str
    temp_path: Path


def list_thermal_zones(base_path: str = "/sys/class/thermal") -> list[ThermalZoneInfo]:
    """
    Discovers readable Linux thermal zones under `base_path`
    (/sys/class/thermal/thermal_zone<N>/{type,temp}). Zones that exist but
    can't currently be read (permission denied, or a `temp` file that's
    momentarily unavailable) are skipped rather than raising -- discovery
    should be best-effort; LinuxThermalZoneReader below decides what to
    do if the resulting list is empty.
    """
    base = Path(base_path)
    if not base.is_dir():
        return []

    zones: list[ThermalZoneInfo] = []
    for entry in sorted(base.glob("thermal_zone*")):
        temp_path = entry / "temp"
        type_path = entry / "type"
        if not temp_path.exists():
            continue
        try:
            index = int(entry.name.removeprefix("thermal_zone"))
        except ValueError:
            continue
        try:
            zone_type = type_path.read_text().strip() if type_path.exists() else "unknown"
        except OSError:
            zone_type = "unknown"
        zones.append(ThermalZoneInfo(index=index, zone_type=zone_type, temp_path=temp_path))
    return zones


class LinuxThermalZoneReader:
    """
    Real hardware telemetry reader for Linux, using
    /sys/class/thermal/thermal_zone*/temp (reported in millidegrees
    Celsius by the kernel).

    This is the fix for STATUS.md's "no real telemetry readers exist"
    gap -- ThermalMonitor's `reader` parameter previously only ever saw
    constants or scripted test sequences. This is a genuine hardware
    reader; pass an instance of it as ThermalMonitor(reader=...).

    Parameters
    ----------
    zone_type_substring:
        If given, only zones whose `type` file contains this substring
        (case-insensitive) are considered -- e.g. "cpu" to avoid reading
        a battery or wifi thermal zone. If None, all discovered zones are
        considered.
    aggregate:
        "max" (default) reports the hottest of the considered zones, which
        is the conservative choice for a throttling decision: any single
        hot zone should be able to trigger throttling. "mean" averages
        them instead.
    base_path:
        Overridable for testing -- point this at a directory built to
        mirror /sys/class/thermal/thermal_zone*/{type,temp}'s layout
        instead of the real sysfs path.

    Raises
    ------
    RuntimeError
        At construction time, if no matching, readable thermal zone can
        be found -- fails fast rather than silently returning a bogus
        temperature (e.g. 0.0) forever at read time, which would be far
        more dangerous for a safety-relevant reader than an explicit
        startup error.
    """

    def __init__(
        self,
        zone_type_substring: Optional[str] = None,
        aggregate: str = "max",
        base_path: str = "/sys/class/thermal",
    ) -> None:
        if aggregate not in ("max", "mean"):
            raise ValueError("aggregate must be 'max' or 'mean'")

        zones = list_thermal_zones(base_path)
        if zone_type_substring is not None:
            needle = zone_type_substring.lower()
            zones = [z for z in zones if needle in z.zone_type.lower()]

        if not zones:
            raise RuntimeError(
                f"no readable thermal zones found under '{base_path}'"
                + (f" matching type substring '{zone_type_substring}'" if zone_type_substring else "")
                + " -- LinuxThermalZoneReader requires real hardware sysfs "
                  "entries; use a ScriptedThermalReader for tests/simulation."
            )

        self._temp_paths = [z.temp_path for z in zones]
        self._aggregate = aggregate

    def __call__(self) -> float:
        readings = []
        for path in self._temp_paths:
            try:
                millidegrees = int(path.read_text().strip())
            except (OSError, ValueError):
                # A zone that was readable at construction time can still
                # transiently fail at read time (race with the kernel
                # updating it, a device being hot-unplugged, etc). Skip it
                # for this sample rather than raising and killing whatever
                # training/inference loop is calling this every step.
                continue
            readings.append(millidegrees / 1000.0)

        if not readings:
            raise RuntimeError(
                "all previously-discovered thermal zones failed to read "
                "on this call -- hardware may have changed since construction"
            )

        if self._aggregate == "max":
            return max(readings)
        return sum(readings) / len(readings)

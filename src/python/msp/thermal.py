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

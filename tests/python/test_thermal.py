import pytest

from msp.thermal import ThermalMonitor, ScriptedThermalReader


def test_default_reader_is_deterministic_and_cool():
    monitor = ThermalMonitor()
    assert monitor.update_ratio() == 1.0
    assert monitor.is_throttling() is False


def test_scripted_reader_drives_throttling_transition():
    reader = ScriptedThermalReader([60.0, 80.0, 90.0])
    monitor = ThermalMonitor(reader=reader, freeze_threshold_c=75.0, freeze_ratio=0.2)

    assert monitor.update_ratio() == 1.0   # 60C, below threshold
    assert monitor.update_ratio() == 0.2   # 80C, throttling
    assert monitor.update_ratio() == 0.2   # 90C, throttling


def test_scripted_reader_holds_last_value_after_exhaustion():
    reader = ScriptedThermalReader([50.0])
    for _ in range(5):
        assert reader() == 50.0


def test_invalid_freeze_ratio_rejected():
    with pytest.raises(ValueError):
        ThermalMonitor(freeze_ratio=0.0)
    with pytest.raises(ValueError):
        ThermalMonitor(freeze_ratio=1.5)


def test_empty_scripted_sequence_rejected():
    with pytest.raises(ValueError):
        ScriptedThermalReader([])

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


# --- LinuxThermalZoneReader tests (simulated sysfs, since this test
# environment has no real thermal zones) ---

from msp.thermal import LinuxThermalZoneReader, list_thermal_zones


def _make_fake_zone(base_dir, index, zone_type, millidegrees):
    zone_dir = base_dir / f"thermal_zone{index}"
    zone_dir.mkdir(parents=True)
    (zone_dir / "type").write_text(zone_type)
    (zone_dir / "temp").write_text(str(millidegrees))
    return zone_dir


def test_list_thermal_zones_discovers_fake_zones(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 45000)
    _make_fake_zone(tmp_path, 1, "battery-thermal", 32000)

    zones = list_thermal_zones(base_path=str(tmp_path))
    assert len(zones) == 2
    assert {z.zone_type for z in zones} == {"cpu-thermal", "battery-thermal"}


def test_list_thermal_zones_on_missing_directory_returns_empty():
    assert list_thermal_zones(base_path="/nonexistent/path/for/sure") == []


def test_linux_reader_converts_millidegrees_to_celsius(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 45123)
    reader = LinuxThermalZoneReader(base_path=str(tmp_path))
    assert reader() == pytest.approx(45.123)


def test_linux_reader_max_aggregate_picks_hottest_zone(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 45000)
    _make_fake_zone(tmp_path, 1, "gpu-thermal", 78000)
    _make_fake_zone(tmp_path, 2, "battery-thermal", 30000)

    reader = LinuxThermalZoneReader(base_path=str(tmp_path), aggregate="max")
    assert reader() == pytest.approx(78.0)


def test_linux_reader_mean_aggregate_averages_zones(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 40000)
    _make_fake_zone(tmp_path, 1, "cpu-thermal", 60000)

    reader = LinuxThermalZoneReader(base_path=str(tmp_path), aggregate="mean")
    assert reader() == pytest.approx(50.0)


def test_linux_reader_filters_by_zone_type_substring(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 40000)
    _make_fake_zone(tmp_path, 1, "gpu-thermal", 90000)  # would dominate max() if not filtered out

    reader = LinuxThermalZoneReader(base_path=str(tmp_path), zone_type_substring="cpu")
    assert reader() == pytest.approx(40.0)


def test_linux_reader_type_filter_is_case_insensitive(tmp_path):
    _make_fake_zone(tmp_path, 0, "CPU-Thermal", 50000)
    reader = LinuxThermalZoneReader(base_path=str(tmp_path), zone_type_substring="cpu")
    assert reader() == pytest.approx(50.0)


def test_linux_reader_raises_at_construction_if_no_zones_found(tmp_path):
    with pytest.raises(RuntimeError):
        LinuxThermalZoneReader(base_path=str(tmp_path))  # empty dir, no zones


def test_linux_reader_raises_if_type_filter_matches_nothing(tmp_path):
    _make_fake_zone(tmp_path, 0, "battery-thermal", 30000)
    with pytest.raises(RuntimeError):
        LinuxThermalZoneReader(base_path=str(tmp_path), zone_type_substring="gpu")


def test_linux_reader_invalid_aggregate_rejected(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 40000)
    with pytest.raises(ValueError):
        LinuxThermalZoneReader(base_path=str(tmp_path), aggregate="median")


def test_linux_reader_skips_unreadable_zone_at_read_time(tmp_path):
    """A zone that becomes unreadable after construction (e.g. a
    momentarily garbled sysfs value) must be skipped for that sample, not
    crash the whole reader, as long as at least one other zone still
    works."""
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 50000)
    zone1_dir = _make_fake_zone(tmp_path, 1, "gpu-thermal", 60000)

    reader = LinuxThermalZoneReader(base_path=str(tmp_path), aggregate="max")
    (zone1_dir / "temp").write_text("not-a-number")  # corrupt it post-construction

    assert reader() == pytest.approx(50.0)  # falls back to the still-good zone


# --- WatchdogTelemetryReader tests: exercises the real compiled
# `watchdogd` binary (src/daemon/watchdogd_main.c) end-to-end -- C main()
# -> thermal_reader.c -> telemetry_server.c -> socket -> this Python
# client. Skipped if the binary hasn't been built, same pattern
# test_adapter_manager.py uses for msp_native. ---

import os
import subprocess
import time

from msp.thermal import WatchdogTelemetryReader


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


@requires_watchdogd
def test_watchdogd_integration_serves_real_thermal_reading(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 61500)  # 61.5C
    socket_path = tmp_path / "watchdogd.sock"

    proc = subprocess.Popen(
        [
            _WATCHDOGD_BIN,
            "--socket", str(socket_path),
            "--thermal-base", str(tmp_path),
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for_socket(str(socket_path)), (
            f"watchdogd never created its socket; stderr so far:\n"
            f"{proc.stderr.read() if proc.poll() is not None else '(still running)'}"
        )

        reader = WatchdogTelemetryReader(socket_path=str(socket_path))
        temp_c, sram_bytes, security_violation = reader.read_full()
        assert temp_c == pytest.approx(61.5, abs=1e-2)
        assert sram_bytes == 0  # thermal reader only ever fills temperature
        assert security_violation is False

        monitor = ThermalMonitor(reader=reader, freeze_threshold_c=75.0)
        assert monitor.is_throttling() is False  # 61.5C is below threshold

        # A second, independent query -- proves it's a live source, not a
        # one-shot/cached response.
        temp_c_again, _, _ = reader.read_full()
        assert temp_c_again == pytest.approx(61.5, abs=1e-2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


@requires_watchdogd
def test_watchdogd_integration_removes_socket_on_graceful_shutdown(tmp_path):
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 40000)
    socket_path = tmp_path / "watchdogd.sock"

    proc = subprocess.Popen(
        [_WATCHDOGD_BIN, "--socket", str(socket_path), "--thermal-base", str(tmp_path)],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for_socket(str(socket_path))
        proc.terminate()  # SIGTERM -- watchdogd's installed graceful-shutdown handler
        assert proc.wait(timeout=5.0) == 0
        assert not socket_path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_linux_reader_integrates_with_thermal_monitor(tmp_path):
    """End-to-end: a real (simulated) hardware reader driving
    ThermalMonitor's throttling decision, not just a ScriptedThermalReader."""
    _make_fake_zone(tmp_path, 0, "cpu-thermal", 90000)  # 90C, hot
    reader = LinuxThermalZoneReader(base_path=str(tmp_path))
    monitor = ThermalMonitor(reader=reader, freeze_threshold_c=75.0)
    assert monitor.is_throttling() is True

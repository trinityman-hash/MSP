// thermal_reader.h
//
// Real Linux /sys/class/thermal reader for the C side of the codebase,
// mirroring src/python/msp/thermal.py's LinuxThermalZoneReader so both
// languages implement the same policy (enumerate thermal_zone* entries,
// optionally filter by a case-insensitive substring of the zone's
// "type" file, aggregate multiple matching zones by max or mean) instead
// of drifting apart.
//
// WHY A SECOND C IMPLEMENTATION RATHER THAN CALLING INTO THE PYTHON ONE:
// embedding a Python interpreter into sandbox_watchdog (e.g. via
// pybind11's embedding mode) was considered and rejected -- the whole
// point of this daemon is to be a small, dependency-light safety
// mechanism that can plausibly keep working even if the main Python
// process is in a bad state; giving it a hard runtime dependency on a
// healthy Python interpreter would undermine that. A second, small,
// independently-testable C implementation of the same sysfs-reading
// policy is the better tradeoff here. If the two ever need to be
// guaranteed identical bit-for-bit, the better fix is a shared spec/test
// vectors file both suites run against, not merging the runtimes.
//
// This module only fills in temperature_c on an msp_telemetry_t --
// sram_usage_bytes and security_violation are left at 0/false by
// msp_thermal_reader_telemetry_fn(), same as the Python side's
// ThermalMonitor is deliberately temperature-only and independent of
// AdapterManager's own SRAM accounting. A real deployment composes this
// reader with a real SRAM/security source rather than expecting one
// reader to know about all three.

#ifndef MSP_THERMAL_READER_H
#define MSP_THERMAL_READER_H

#include "sandbox_watchdog.h"  // for msp_telemetry_t / msp_telemetry_reader_fn

#ifdef __cplusplus
extern "C" {
#endif

#define MSP_THERMAL_READER_BASE_PATH_MAX 256
#define MSP_THERMAL_READER_TYPE_FILTER_MAX 64

typedef enum {
    MSP_THERMAL_AGGREGATE_MAX,
    MSP_THERMAL_AGGREGATE_MEAN,
} msp_thermal_aggregate_t;

typedef struct {
    char base_path[MSP_THERMAL_READER_BASE_PATH_MAX];
    char zone_type_filter[MSP_THERMAL_READER_TYPE_FILTER_MAX];  // substring, case-insensitive; empty = no filter
    msp_thermal_aggregate_t aggregate;
} msp_thermal_reader_config_t;

// Fills `config` with the production defaults: base_path =
// "/sys/class/thermal", no zone-type filter, aggregate = MAX. Tests
// override base_path (and optionally zone_type_filter/aggregate)
// afterwards to point at a simulated sysfs tree.
void msp_thermal_reader_default_config(msp_thermal_reader_config_t* config);

// Reads real thermal zone temperatures under config->base_path, keeping
// only zones whose "type" file contains config->zone_type_filter as a
// case-insensitive substring (all zones, if the filter is empty), and
// aggregates the matching zones' temperatures per config->aggregate.
//
// A zone that exists but can't be read cleanly (missing/unreadable
// "temp" file, non-numeric contents) is skipped for that call rather
// than failing the whole read, as long as at least one other matching
// zone is readable -- mirrors the Python reader's per-zone error
// tolerance (see test_linux_reader_skips_unreadable_zone_at_read_time in
// tests/python/test_thermal.py for the equivalent Python behavior this
// is matching).
//
// Returns 0 on success, with *out_temp_c set to the aggregated
// temperature in Celsius. Returns -1 if config->base_path doesn't exist,
// or no zone both matched the filter and was readable -- *out_temp_c is
// left untouched in that case.
int msp_thermal_reader_read(const msp_thermal_reader_config_t* config,
                             float* out_temp_c);

// msp_telemetry_reader_fn-compatible adapter (see sandbox_watchdog.h):
// `user_data` must point to a msp_thermal_reader_config_t. On a
// successful read, out->temperature_c is the real aggregated
// temperature. On a FAILED read (see msp_thermal_reader_read above),
// out->temperature_c is set to +INFINITY rather than left at some
// default like 0 -- 0 would read as "very cold" to the watchdog, which
// is the opposite of what a broken sensor path should imply for a safety
// mechanism. +INFINITY guarantees a read failure can only ever cause the
// watchdog to over-trigger, never to silently miss a real violation
// because the failure looked identical to a cold, safe reading.
void msp_thermal_reader_telemetry_fn(msp_telemetry_t* out, void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // MSP_THERMAL_READER_H

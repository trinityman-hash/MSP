// Tests for src/daemon/thermal_reader.c, against a simulated sysfs tree
// (this test environment, like the Python suite's, has no real thermal
// zones to read). Deliberately mirrors tests/python/test_thermal.py's
// LinuxThermalZoneReader cases so the two implementations are checked
// against the same behavior, not just each compiling cleanly.

#include "thermal_reader.h"

#include <errno.h>
#include <ftw.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAILED: %s (line %d)\n", #cond, __LINE__);    \
            return 1;                                                      \
        }                                                                  \
    } while (0)

#define CHECK_APPROX(a, b)                                                 \
    do {                                                                   \
        double diff = fabs((double)(a) - (double)(b));                    \
        if (diff > 1e-3) {                                                 \
            fprintf(stderr, "FAILED: %s (%.4f) != %s (%.4f) (line %d)\n",  \
                    #a, (double)(a), #b, (double)(b), __LINE__);           \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// --- simulated sysfs tree helpers ---

static void make_fake_zone(const char* base_dir, int index,
                            const char* zone_type, long millidegrees) {
    char zone_dir[300];
    snprintf(zone_dir, sizeof(zone_dir), "%s/thermal_zone%d", base_dir, index);
    mkdir(zone_dir, 0755);

    char type_path[340];
    char temp_path[340];
    snprintf(type_path, sizeof(type_path), "%s/type", zone_dir);
    snprintf(temp_path, sizeof(temp_path), "%s/temp", zone_dir);

    FILE* tf = fopen(type_path, "w");
    fprintf(tf, "%s", zone_type);
    fclose(tf);

    FILE* pf = fopen(temp_path, "w");
    fprintf(pf, "%ld", millidegrees);
    fclose(pf);
}

static void corrupt_zone_temp(const char* base_dir, int index) {
    char temp_path[340];
    snprintf(temp_path, sizeof(temp_path), "%s/thermal_zone%d/temp", base_dir, index);
    FILE* pf = fopen(temp_path, "w");
    fprintf(pf, "not-a-number");
    fclose(pf);
}

static int nftw_remove_cb(const char* path, const struct stat* sb,
                           int typeflag, struct FTW* ftwbuf) {
    (void)sb;
    (void)typeflag;
    (void)ftwbuf;
    return remove(path);
}

static void remove_tree(const char* path) {
    nftw(path, nftw_remove_cb, 16, FTW_DEPTH | FTW_PHYS);
}

static int make_temp_dir(char* out, size_t out_len) {
    snprintf(out, out_len, "/tmp/msp_thermal_test_XXXXXX");
    return mkdtemp(out) != NULL ? 0 : -1;
}

// --- tests ---

static int test_default_config_matches_production_path(void) {
    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    CHECK(strcmp(config.base_path, "/sys/class/thermal") == 0);
    CHECK(config.zone_type_filter[0] == '\0');
    CHECK(config.aggregate == MSP_THERMAL_AGGREGATE_MAX);
    return 0;
}

static int test_converts_millidegrees_to_celsius(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "cpu-thermal", 45123);

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == 0);
    CHECK_APPROX(temp_c, 45.123);

    remove_tree(dir);
    return 0;
}

static int test_max_aggregate_picks_hottest_zone(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "cpu-thermal", 45000);
    make_fake_zone(dir, 1, "gpu-thermal", 78000);
    make_fake_zone(dir, 2, "battery-thermal", 30000);

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);
    config.aggregate = MSP_THERMAL_AGGREGATE_MAX;

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == 0);
    CHECK_APPROX(temp_c, 78.0);

    remove_tree(dir);
    return 0;
}

static int test_mean_aggregate_averages_zones(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "cpu-thermal", 40000);
    make_fake_zone(dir, 1, "cpu-thermal", 60000);

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);
    config.aggregate = MSP_THERMAL_AGGREGATE_MEAN;

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == 0);
    CHECK_APPROX(temp_c, 50.0);

    remove_tree(dir);
    return 0;
}

static int test_filters_by_zone_type_substring(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "cpu-thermal", 40000);
    make_fake_zone(dir, 1, "gpu-thermal", 90000);  // would dominate max() if not filtered out

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);
    snprintf(config.zone_type_filter, sizeof(config.zone_type_filter), "cpu");

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == 0);
    CHECK_APPROX(temp_c, 40.0);

    remove_tree(dir);
    return 0;
}

static int test_type_filter_is_case_insensitive(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "CPU-Thermal", 50000);

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);
    snprintf(config.zone_type_filter, sizeof(config.zone_type_filter), "cpu");

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == 0);
    CHECK_APPROX(temp_c, 50.0);

    remove_tree(dir);
    return 0;
}

static int test_returns_error_if_no_zones_found(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);  // empty dir, no zones

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == -1);

    remove_tree(dir);
    return 0;
}

static int test_returns_error_if_type_filter_matches_nothing(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "battery-thermal", 30000);

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);
    snprintf(config.zone_type_filter, sizeof(config.zone_type_filter), "gpu");

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == -1);

    remove_tree(dir);
    return 0;
}

static int test_returns_error_for_nonexistent_base_path(void) {
    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "/nonexistent/path/for/sure");

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == -1);
    return 0;
}

static int test_skips_unreadable_zone_falls_back_to_good_one(void) {
    // A zone that becomes unreadable after being set up (e.g. a
    // momentarily garbled sysfs value) must be skipped for that read,
    // not crash or fail the whole read, as long as another matching zone
    // is still readable -- matches
    // test_linux_reader_skips_unreadable_zone_at_read_time in the Python
    // suite.
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "cpu-thermal", 50000);
    make_fake_zone(dir, 1, "gpu-thermal", 60000);
    corrupt_zone_temp(dir, 1);

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);
    config.aggregate = MSP_THERMAL_AGGREGATE_MAX;

    float temp_c = 0.0f;
    CHECK(msp_thermal_reader_read(&config, &temp_c) == 0);
    CHECK_APPROX(temp_c, 50.0);  // falls back to the still-good zone

    remove_tree(dir);
    return 0;
}

static int test_telemetry_fn_success_sets_temperature_and_zeroes_rest(void) {
    char dir[64];
    CHECK(make_temp_dir(dir, sizeof(dir)) == 0);
    make_fake_zone(dir, 0, "cpu-thermal", 90000);  // 90C, hot

    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "%s", dir);

    msp_telemetry_t telemetry;
    memset(&telemetry, 0xAB, sizeof(telemetry));  // poison, so zeroing is actually verified
    msp_thermal_reader_telemetry_fn(&telemetry, &config);

    CHECK_APPROX(telemetry.temperature_c, 90.0);
    CHECK(telemetry.sram_usage_bytes == 0);
    CHECK(telemetry.security_violation == false);

    remove_tree(dir);
    return 0;
}

static int test_telemetry_fn_failure_reports_infinity_not_zero(void) {
    // Critical safety property: a broken sensor path must fail HOT
    // (trips the watchdog), never silently read as 0C ("cold" = safe) --
    // see thermal_reader.h for the full rationale.
    msp_thermal_reader_config_t config;
    msp_thermal_reader_default_config(&config);
    snprintf(config.base_path, sizeof(config.base_path), "/nonexistent/path/for/sure");

    msp_telemetry_t telemetry;
    memset(&telemetry, 0, sizeof(telemetry));
    msp_thermal_reader_telemetry_fn(&telemetry, &config);

    CHECK(isinf(telemetry.temperature_c) && telemetry.temperature_c > 0);
    CHECK(telemetry.temperature_c > MSP_CRITICAL_TEMP_C);  // would actually trip the watchdog
    return 0;
}

int main(void) {
    int failures = 0;
#define RUN(fn)                              \
    do {                                     \
        if (fn() != 0) {                     \
            fprintf(stderr, #fn " failed\n"); \
            failures++;                      \
        }                                    \
    } while (0)

    RUN(test_default_config_matches_production_path);
    RUN(test_converts_millidegrees_to_celsius);
    RUN(test_max_aggregate_picks_hottest_zone);
    RUN(test_mean_aggregate_averages_zones);
    RUN(test_filters_by_zone_type_substring);
    RUN(test_type_filter_is_case_insensitive);
    RUN(test_returns_error_if_no_zones_found);
    RUN(test_returns_error_if_type_filter_matches_nothing);
    RUN(test_returns_error_for_nonexistent_base_path);
    RUN(test_skips_unreadable_zone_falls_back_to_good_one);
    RUN(test_telemetry_fn_success_sets_temperature_and_zeroes_rest);
    RUN(test_telemetry_fn_failure_reports_infinity_not_zero);

#undef RUN

    if (failures > 0) {
        fprintf(stderr, "%d test(s) failed.\n", failures);
        return 1;
    }
    printf("All thermal_reader tests passed.\n");
    return 0;
}

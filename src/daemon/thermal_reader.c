// thermal_reader.c
// See thermal_reader.h for the design rationale.

#include "thermal_reader.h"

#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

void msp_thermal_reader_default_config(msp_thermal_reader_config_t* config) {
    memset(config, 0, sizeof(*config));
    snprintf(config->base_path, sizeof(config->base_path), "/sys/class/thermal");
    config->zone_type_filter[0] = '\0';
    config->aggregate = MSP_THERMAL_AGGREGATE_MAX;
}

static int read_zone_type(const char* zone_dir, char* out, size_t out_len) {
    char path[320];
    int n = snprintf(path, sizeof(path), "%s/type", zone_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        return -1;
    }
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    if (!fgets(out, (int)out_len, f)) {
        fclose(f);
        return -1;
    }
    fclose(f);
    size_t len = strlen(out);
    while (len > 0 && (out[len - 1] == '\n' || out[len - 1] == '\r')) {
        out[--len] = '\0';
    }
    return 0;
}

static int read_zone_millidegrees(const char* zone_dir, long* out_millideg) {
    char path[320];
    int n = snprintf(path, sizeof(path), "%s/temp", zone_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        return -1;
    }
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    long value;
    int matched = fscanf(f, "%ld", &value);
    fclose(f);
    if (matched != 1) {
        return -1;
    }
    *out_millideg = value;
    return 0;
}

// Case-insensitive substring search (no libc guarantee of strcasestr
// availability/portability across libc's, so a small local
// implementation instead of relying on it).
static int contains_ci(const char* haystack, const char* needle) {
    if (needle[0] == '\0') {
        return 1;  // empty filter matches everything
    }
    size_t hn = strlen(haystack);
    size_t nn = strlen(needle);
    if (nn > hn) {
        return 0;
    }
    for (size_t i = 0; i + nn <= hn; i++) {
        size_t j = 0;
        for (; j < nn; j++) {
            if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j])) {
                break;
            }
        }
        if (j == nn) {
            return 1;
        }
    }
    return 0;
}

int msp_thermal_reader_read(const msp_thermal_reader_config_t* config,
                             float* out_temp_c) {
    DIR* dir = opendir(config->base_path);
    if (!dir) {
        return -1;
    }

    double sum_c = 0.0;
    double max_c = -HUGE_VAL;
    long count = 0;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "thermal_zone", 12) != 0) {
            continue;
        }

        char zone_dir[300];
        int n = snprintf(zone_dir, sizeof(zone_dir), "%s/%s", config->base_path, entry->d_name);
        if (n < 0 || (size_t)n >= sizeof(zone_dir)) {
            continue;
        }

        char zone_type[MSP_THERMAL_READER_TYPE_FILTER_MAX];
        if (read_zone_type(zone_dir, zone_type, sizeof(zone_type)) != 0) {
            // Matches the Python reader's fallback: an unreadable "type"
            // file doesn't disqualify the zone outright, it just means
            // it won't match any non-empty filter.
            strncpy(zone_type, "unknown", sizeof(zone_type) - 1);
            zone_type[sizeof(zone_type) - 1] = '\0';
        }
        if (!contains_ci(zone_type, config->zone_type_filter)) {
            continue;
        }

        long millideg;
        if (read_zone_millidegrees(zone_dir, &millideg) != 0) {
            continue;  // unreadable/corrupt "temp" for this sample: skip, don't fail the whole read
        }

        double c = (double)millideg / 1000.0;
        sum_c += c;
        count++;
        if (c > max_c) {
            max_c = c;
        }
    }
    closedir(dir);

    if (count == 0) {
        return -1;
    }

    *out_temp_c = (float)(config->aggregate == MSP_THERMAL_AGGREGATE_MEAN
                               ? (sum_c / (double)count)
                               : max_c);
    return 0;
}

void msp_thermal_reader_telemetry_fn(msp_telemetry_t* out, void* user_data) {
    const msp_thermal_reader_config_t* config = (const msp_thermal_reader_config_t*)user_data;
    float temp_c;
    if (msp_thermal_reader_read(config, &temp_c) == 0) {
        out->temperature_c = temp_c;
    } else {
        out->temperature_c = INFINITY;  // see header: fail hot, not cold
    }
    out->sram_usage_bytes = 0;
    out->security_violation = false;
}

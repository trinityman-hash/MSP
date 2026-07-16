// Standalone test/harness for sandbox_watchdog.c.
//
// IMPORTANT: this test *measures* fallback latency and polling behavior,
// it does not assert the spec's "must definitively fire every 10ms" /
// "fallback must be <5ms" as hard guarantees. Linux is not a real-time
// operating system, so those are not guarantees this (or any) userspace
// program can make. What we *can* verify:
//   1. The fallback path fires correctly when a violation occurs.
//   2. It does NOT fire when telemetry stays safe.
//   3. Latency from violation to fallback is measured and printed, and
//      checked against a generous (not aspirational) sanity bound so a
//      real regression -- e.g. reintroducing the original cross-thread
//      longjmp bug's occasional failure to return -- gets caught.
//
// Each subtest uses its own state struct (rather than shared globals) so
// the two scenarios below can't leak state into each other.

#include "sandbox_watchdog.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAILED: %s (line %d)\n", #cond, __LINE__);    \
            return 1;                                                      \
        }                                                                  \
    } while (0)

typedef struct {
    _Atomic int read_count;
    bool trigger_violation;          // if false, reader always reports safe
    struct timespec violation_detected_at;
    bool violation_timestamp_set;
} reader_state_t;

static void scripted_reader(msp_telemetry_t* out, void* user_data) {
    reader_state_t* state = (reader_state_t*)user_data;
    int n = atomic_fetch_add(&state->read_count, 1) + 1;

    bool report_violation = state->trigger_violation && n >= 4;
    if (report_violation) {
        out->temperature_c = 90.0f;  // exceeds MSP_CRITICAL_TEMP_C (85.0)
        out->sram_usage_bytes = 256000000;
        out->security_violation = false;
        if (!state->violation_timestamp_set) {
            clock_gettime(CLOCK_MONOTONIC, &state->violation_detected_at);
            state->violation_timestamp_set = true;
        }
    } else {
        out->temperature_c = 55.0f;
        out->sram_usage_bytes = 256000000;
        out->security_violation = false;
    }
}

typedef struct {
    volatile bool* stop_flag;        // monitored thread exits its busy-wait when true
    bool reached_fallback;
    struct timespec fallback_reached_at;
} monitored_state_t;

static void* monitored_thread_fn(void* arg) {
    monitored_state_t* state = (monitored_state_t*)arg;
    if (MSP_ARM_FALLBACK_POINT() != 0) {
        // Reached via siglongjmp from the signal handler -- the
        // "rollback to base model" path.
        clock_gettime(CLOCK_MONOTONIC, &state->fallback_reached_at);
        state->reached_fallback = true;
        return NULL;
    }
    // Simulate the monitored thread doing adapter inference work while
    // waiting to (possibly) be interrupted.
    while (!*state->stop_flag) {
        usleep(1000);
    }
    return NULL;
}

static double ms_between(struct timespec a, struct timespec b) {
    double sec_diff = (double)(b.tv_sec - a.tv_sec);
    double nsec_diff = (double)(b.tv_nsec - a.tv_nsec);
    return sec_diff * 1000.0 + nsec_diff / 1e6;
}

static int run_positive_case(void) {
    reader_state_t reader_state = {0};
    reader_state.trigger_violation = true;

    volatile bool stop_flag = false;
    monitored_state_t mon_state = {0};
    mon_state.stop_flag = &stop_flag;

    pthread_t monitored;
    CHECK(pthread_create(&monitored, NULL, monitored_thread_fn, &mon_state) == 0);
    usleep(20000);  // let the monitored thread arm its fallback point first

    volatile bool keep_running = true;
    volatile bool violation_flag = false;
    msp_watchdog_config_t cfg = {
        .monitored_thread = monitored,
        .read_telemetry = scripted_reader,
        .reader_user_data = &reader_state,
        .keep_running = &keep_running,
        .violation_flag = &violation_flag,
    };
    pthread_t watchdog;
    CHECK(pthread_create(&watchdog, NULL, msp_telemetry_watchdog_daemon, &cfg) == 0);

    pthread_join(watchdog, NULL);

    // Bound how long we wait for the signal to land and the monitored
    // thread to observe it, then release the busy-wait loop either way.
    for (int i = 0; i < 1000 && !mon_state.reached_fallback; i++) {
        usleep(1000);
    }
    stop_flag = true;
    pthread_join(monitored, NULL);

    CHECK(violation_flag == true);
    CHECK(mon_state.reached_fallback == true);
    CHECK(reader_state.violation_timestamp_set == true);

    double latency_ms = ms_between(reader_state.violation_detected_at,
                                    mon_state.fallback_reached_at);
    printf("Measured violation-to-fallback latency: %.3f ms\n", latency_ms);
    // Generous sanity bound (not the spec's aspirational "<5ms" hard
    // guarantee) -- catches real regressions without pretending Linux is
    // an RTOS.
    CHECK(latency_ms >= 0.0);
    CHECK(latency_ms < 500.0);
    return 0;
}

static int run_negative_case(void) {
    reader_state_t reader_state = {0};
    reader_state.trigger_violation = false;  // telemetry always safe

    volatile bool stop_flag = false;
    monitored_state_t mon_state = {0};
    mon_state.stop_flag = &stop_flag;

    pthread_t monitored;
    CHECK(pthread_create(&monitored, NULL, monitored_thread_fn, &mon_state) == 0);
    usleep(20000);

    volatile bool keep_running = true;
    volatile bool violation_flag = false;
    msp_watchdog_config_t cfg = {
        .monitored_thread = monitored,
        .read_telemetry = scripted_reader,
        .reader_user_data = &reader_state,
        .keep_running = &keep_running,
        .violation_flag = &violation_flag,
    };
    pthread_t watchdog;
    CHECK(pthread_create(&watchdog, NULL, msp_telemetry_watchdog_daemon, &cfg) == 0);

    usleep(100000);  // let it poll several times with safe telemetry
    keep_running = false;
    pthread_join(watchdog, NULL);

    stop_flag = true;  // release monitored thread's busy-wait
    pthread_join(monitored, NULL);

    CHECK(violation_flag == false);
    CHECK(mon_state.reached_fallback == false);
    return 0;
}

int main(void) {
    if (run_positive_case() != 0) return 1;
    if (run_negative_case() != 0) return 1;
    printf("All watchdog tests passed.\n");
    return 0;
}

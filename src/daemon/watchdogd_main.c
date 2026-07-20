// watchdogd_main.c
//
// The standalone daemon entry point STATUS.md's "What's left to do" #2
// flagged as missing: "thermal_reader.c ... is a drop-in fit for
// msp_watchdog_config_t's read_telemetry field -- no daemon entry point
// actually constructs one with it by default yet."
//
// What this process does, concretely:
//   1. Builds a msp_thermal_reader_config_t (real /sys/class/thermal by
//      default; overridable below for testing against a simulated tree).
//   2. Runs a telemetry Unix-domain-socket server (telemetry_server.c) as
//      its "protected work" -- the thing a violation rolls back -- using
//      msp_thermal_reader_telemetry_fn as the reader. This is also the
//      shared data source that closes the second half of the gap:
//      msp.thermal.WatchdogTelemetryReader (Python) queries this same
//      socket, so Python and the watchdog now read one real telemetry
//      source instead of two independent guesses at /sys/class/thermal
//      agreeing.
//   3. Runs msp_telemetry_watchdog_daemon (sandbox_watchdog.c) concurrently
//      against that same reader, watching for a real violation.
//   4. On violation, the watchdog's signal interrupts step 2's blocking
//      work via siglongjmp (see sandbox_watchdog.h) back to this file's
//      own MSP_ARM_FALLBACK_POINT() call -- not a crash, a controlled
//      rollback. This process then re-arms and resumes serving telemetry
//      after a short cooldown, demonstrating the re-arm pattern
//      sandbox_watchdog.h's fix #4 and STATUS.md both call out as the
//      real deployment's responsibility ("a real deployment would
//      typically re-arm after the monitored thread recovers").
//
// This process's own "protected work" (serving telemetry) is deliberately
// modest -- there is no adapter inference loop to protect yet (see
// STATUS.md item 3, "end-to-end training example"). What's demonstrated
// and tested here is the wiring itself: a real telemetry source driving a
// real watchdog with a real, working fallback/re-arm cycle, which a
// future inference-serving process can reuse by arming its OWN fallback
// point around ITS OWN protected work instead of (or alongside) this
// process's socket loop.

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sandbox_watchdog.h"
#include "telemetry_server.h"
#include "thermal_reader.h"

static volatile bool g_keep_running = true;

static void handle_shutdown_signal(int signo) {
    (void)signo;
    // Async-signal-safe: setting a sig_atomic_t/bool flag is the
    // canonical safe operation inside a signal handler. The main loop
    // below observes it within one MSP_TELEMETRY_SERVER_POLL_INTERVAL_US
    // / MSP_WATCHDOG_POLL_INTERVAL_US tick.
    g_keep_running = false;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "\n"
            "Reference watchdog daemon: watches real thermal telemetry and\n"
            "serves it to any client (see msp.thermal.WatchdogTelemetryReader\n"
            "on the Python side) over a Unix domain socket.\n"
            "\n"
            "Options:\n"
            "  --socket PATH        Unix domain socket to serve telemetry on.\n"
            "                       Default: $MSP_WATCHDOG_SOCKET, or\n"
            "                       /tmp/msp_watchdogd.sock if unset.\n"
            "  --thermal-base PATH  Base dir to read thermal zones from.\n"
            "                       Default: /sys/class/thermal. Override this\n"
            "                       to point at a simulated sysfs tree for\n"
            "                       testing (see tests/daemon and\n"
            "                       tests/python/test_thermal.py for real\n"
            "                       examples of building one).\n"
            "  --zone-type STRING   Only consider zones whose type contains\n"
            "                       this substring (case-insensitive).\n"
            "                       Default: no filter, all zones considered.\n"
            "  --aggregate max|mean How to combine multiple matching zones.\n"
            "                       Default: max.\n"
            "  -h, --help           Show this help and exit.\n",
            prog);
}

int main(int argc, char** argv) {
    const char* socket_path = getenv("MSP_WATCHDOG_SOCKET");
    if (!socket_path || socket_path[0] == '\0') {
        socket_path = "/tmp/msp_watchdogd.sock";
    }

    msp_thermal_reader_config_t thermal_cfg;
    msp_thermal_reader_default_config(&thermal_cfg);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "--thermal-base") == 0 && i + 1 < argc) {
            snprintf(thermal_cfg.base_path, sizeof(thermal_cfg.base_path), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--zone-type") == 0 && i + 1 < argc) {
            snprintf(thermal_cfg.zone_type_filter, sizeof(thermal_cfg.zone_type_filter), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--aggregate") == 0 && i + 1 < argc) {
            const char* val = argv[++i];
            if (strcmp(val, "max") == 0) {
                thermal_cfg.aggregate = MSP_THERMAL_AGGREGATE_MAX;
            } else if (strcmp(val, "mean") == 0) {
                thermal_cfg.aggregate = MSP_THERMAL_AGGREGATE_MEAN;
            } else {
                fprintf(stderr, "--aggregate must be 'max' or 'mean', got '%s'\n", val);
                return 2;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unrecognized argument: '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (strlen(socket_path) >= MSP_TELEMETRY_SERVER_SOCKET_PATH_MAX) {
        fprintf(stderr, "--socket path too long (max %d bytes): '%s'\n",
                MSP_TELEMETRY_SERVER_SOCKET_PATH_MAX - 1, socket_path);
        return 2;
    }

    // A client disconnecting mid-response would otherwise deliver SIGPIPE
    // to this process on the next write(), whose default disposition is
    // to terminate the whole daemon -- clearly wrong for what should be
    // an isolated, per-connection failure. write_all() in
    // telemetry_server.c already handles the resulting EPIPE from
    // write() as an ordinary error; this just stops the signal from
    // killing the process before write() gets the chance to return it.
    signal(SIGPIPE, SIG_IGN);

    // Graceful shutdown on Ctrl-C / `kill` / systemd stop, rather than an
    // abrupt process kill that would leave the socket file behind.
    struct sigaction shutdown_sa;
    memset(&shutdown_sa, 0, sizeof(shutdown_sa));
    shutdown_sa.sa_handler = handle_shutdown_signal;
    sigemptyset(&shutdown_sa.sa_mask);
    shutdown_sa.sa_flags = 0;
    sigaction(SIGINT, &shutdown_sa, NULL);
    sigaction(SIGTERM, &shutdown_sa, NULL);

    fprintf(stderr,
            "[WATCHDOGD] starting: socket=%s thermal_base=%s zone_type_filter=%s aggregate=%s\n",
            socket_path, thermal_cfg.base_path,
            thermal_cfg.zone_type_filter[0] ? thermal_cfg.zone_type_filter : "(none)",
            thermal_cfg.aggregate == MSP_THERMAL_AGGREGATE_MEAN ? "mean" : "max");

    // Each pass through this loop re-arms the fallback point (per
    // sandbox_watchdog.h fix #4: MSP_ARM_FALLBACK_POINT() must expand
    // directly in the function whose frame should be restored to, which
    // is exactly this loop body in main() -- not a helper function).
    while (g_keep_running) {
        volatile bool violation_flag = false;

        if (MSP_ARM_FALLBACK_POINT() != 0) {
            // Reached via siglongjmp from the SIGUSR1 handler: the
            // watchdog thread (started below) detected a real violation
            // and signaled this thread while it was blocked inside
            // msp_telemetry_server_run(). That thread has already exited
            // (msp_telemetry_watchdog_daemon fires the signal at most
            // once per call, then returns -- see sandbox_watchdog.c) and
            // was created detached, so there is nothing to join here.
            fprintf(stderr,
                    "[WATCHDOGD] fallback triggered -- telemetry server "
                    "interrupted by a critical violation. Cooling down "
                    "before resuming.\n");
            sleep(1);  // fixed cooldown; a deployment with a real
                       // inference workload to protect might instead
                       // wait for telemetry to actually clear the
                       // threshold before resuming it -- there is none
                       // here, so a fixed delay is the honest choice
                       // rather than faking a more sophisticated policy.
            continue;
        }

        volatile bool watchdog_keep_running = true;
        msp_watchdog_config_t watchdog_cfg = {
            .monitored_thread = pthread_self(),
            .read_telemetry = msp_thermal_reader_telemetry_fn,
            .reader_user_data = &thermal_cfg,
            .keep_running = &watchdog_keep_running,
            .violation_flag = &violation_flag,
        };
        pthread_t watchdog_thread;
        if (pthread_create(&watchdog_thread, NULL, msp_telemetry_watchdog_daemon, &watchdog_cfg) != 0) {
            fprintf(stderr, "[WATCHDOGD] failed to start watchdog thread: %s\n", strerror(errno));
            return 1;
        }
        // Detached, not joined: on the normal (non-violation) exit path
        // below, watchdog_keep_running is cleared and this thread exits
        // within one poll tick on its own; on the violation path, control
        // never reaches past msp_telemetry_server_run() below at all
        // (siglongjmp jumps straight back to the check above), so a
        // pthread_join() placed after it would simply never run. Detaching
        // makes both exits equally clean without needing two different
        // cleanup paths.
        pthread_detach(watchdog_thread);

        msp_telemetry_server_config_t server_cfg;
        memset(&server_cfg, 0, sizeof(server_cfg));
        snprintf(server_cfg.socket_path, sizeof(server_cfg.socket_path), "%s", socket_path);
        server_cfg.read_telemetry = msp_thermal_reader_telemetry_fn;
        server_cfg.reader_user_data = &thermal_cfg;
        server_cfg.keep_running = &g_keep_running;

        fprintf(stderr, "[WATCHDOGD] serving telemetry on %s\n", socket_path);

        // Blocks here -- serving telemetry queries -- until either
        // g_keep_running goes false (graceful shutdown, the outer while
        // loop then exits too) or this thread is interrupted by
        // MSP_FALLBACK_SIGNAL (a violation, handled via siglongjmp back
        // to the MSP_ARM_FALLBACK_POINT() check above, NOT by returning
        // here).
        msp_telemetry_server_run(&server_cfg);

        // Only reached via graceful shutdown. Ask the watchdog thread to
        // wind down too, so a fast Ctrl-C doesn't race a stale watchdog
        // thread against process exit (harmless either way since it's
        // detached, but cleaner to ask it to stop).
        watchdog_keep_running = false;
    }

    fprintf(stderr, "[WATCHDOGD] shutting down.\n");
    return 0;
}

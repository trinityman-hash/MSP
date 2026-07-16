// sandbox_watchdog.c
// See sandbox_watchdog.h for the full rationale behind the fixes here.

#include "sandbox_watchdog.h"

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// sigsetjmp buffer for the monitored thread, per fix #4 in the header:
// this MUST be reachable from the macro that expands inline in the
// monitored thread's own function, not hidden behind a function call.
// __thread storage means each monitored thread gets its own independent
// copy, so multiple monitored threads can each arm their own fallback
// point safely.
__thread sigjmp_buf msp_fallback_env;
__thread volatile sig_atomic_t msp_fallback_armed = 0;

static void fallback_signal_handler(int signo) {
    (void)signo;
    if (msp_fallback_armed) {
        // Async-signal-safe: siglongjmp is on the POSIX async-signal-safe
        // list. This jumps within the SAME thread that received the
        // signal, to a sigsetjmp call that is still live on that thread's
        // own stack (armed via the MSP_ARM_FALLBACK_POINT() macro directly
        // in the monitored function) -- well-defined, unlike both the
        // spec's original cross-thread longjmp and this fix's own first
        // draft (a wrapper function that had already returned).
        siglongjmp(msp_fallback_env, 1);
    }
    // If not armed, there's nothing safe to do but return; the caller
    // never reached MSP_ARM_FALLBACK_POINT().
}

int msp_install_fallback_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = fallback_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // deliberately no SA_RESTART: we want the signal to
                       // interrupt whatever the monitored thread is doing.
    if (sigaction(MSP_FALLBACK_SIGNAL, &sa, NULL) != 0) {
        return -1;
    }
    msp_fallback_armed = 1;
    return 0;
}

void* msp_allocate_isolated_region(size_t size) {
    if (size > MSP_MAX_SRAM_ALLOC) {
        return NULL;
    }
    // NO PROT_EXEC: data mapped here cannot be directly executed. This is
    // a real, useful property -- see the header comment for what it does
    // and does NOT provide.
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }
    return ptr;
}

void* msp_telemetry_watchdog_daemon(void* arg) {
    msp_watchdog_config_t* cfg = (msp_watchdog_config_t*)arg;

    while (*cfg->keep_running) {
        usleep(MSP_WATCHDOG_POLL_INTERVAL_US);

        msp_telemetry_t telemetry;
        cfg->read_telemetry(&telemetry, cfg->reader_user_data);

        bool violated = (telemetry.temperature_c > MSP_CRITICAL_TEMP_C) ||
                         (telemetry.sram_usage_bytes > MSP_MAX_SRAM_ALLOC) ||
                         telemetry.security_violation;

        if (violated) {
            fprintf(stderr,
                    "[WATCHDOG] Critical violation detected "
                    "(temp=%.1fC sram=%zu security_violation=%d). "
                    "Signaling fallback.\n",
                    telemetry.temperature_c, telemetry.sram_usage_bytes,
                    telemetry.security_violation);
            if (cfg->violation_flag) {
                *cfg->violation_flag = true;
            }
            // Signal, don't jump directly -- see header for why.
            pthread_kill(cfg->monitored_thread, MSP_FALLBACK_SIGNAL);
            // Stop polling after triggering fallback once; a real
            // deployment would typically re-arm after the monitored
            // thread recovers by calling MSP_ARM_FALLBACK_POINT() again.
            break;
        }
    }
    return NULL;
}

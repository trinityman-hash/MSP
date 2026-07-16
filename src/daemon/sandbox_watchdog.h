// sandbox_watchdog.h
//
// Fixes relative to the v3.0 spec's sandbox_watchdog.c:
//
// 1. UNDEFINED BEHAVIOR (the critical bug): the spec saved a jmp_buf on
//    one thread (the adapter/inference thread) and called longjmp() to it
//    from a *different* thread (the watchdog thread). POSIX setjmp/longjmp
//    are only well-defined for returning to a point earlier in the SAME
//    thread's call stack -- jumping across threads corrupts the target
//    thread's stack and is a use-after-scope bug that "usually" appears to
//    work until it doesn't (typically under load, which is exactly when a
//    thermal/security fallback needs to actually work).
//
//    Fix: the monitored thread installs a signal handler and calls
//    sigsetjmp() itself; the watchdog thread signals it via pthread_kill()
//    instead of jumping directly. siglongjmp() runs inside the signal
//    handler, in the interrupted thread's own context, which is
//    well-defined. This preserves the spec's goal (fast, watchdog-driven
//    rollback to the base model) without the UB.
//
// 2. "RSA-4096 hash validation": RSA is a signature/encryption primitive,
//    not a hash function -- the spec's comment conflated the two. Fix:
//    integrity is checked with a real hash (SHA-256, via OpenSSL) in
//    integrity_check.c. If authenticity (not just integrity) of a plugin
//    is required, that hash should additionally be verified against an
//    RSA or Ed25519 *signature* -- a distinct, separate step, sketched
//    in integrity_check.h.
//
// 3. "Must definitively fire every 10ms" is not achievable as a hard
//    guarantee on a general-purpose (non-RTOS) OS like Linux -- the
//    scheduler can be preempted by higher-priority work. This
//    implementation polls at a 10ms *target* interval and the test suite
//    measures actual jitter instead of asserting an impossible guarantee.
//    See tests/daemon for the measurement harness.

// 4. A SECOND, MORE SUBTLE BUG FOUND DURING TESTING OF THIS FIX: the first
//    draft of this fix exposed arming as a plain function,
//    `msp_arm_fallback_point()`, which did `return sigsetjmp(env, 1);`.
//    That is broken for exactly the same underlying reason as the
//    original spec's bug (#1 above): sigsetjmp/setjmp save the stack
//    pointer as of the call, but once that wrapper function *returns*
//    (the normal, first return with value 0), its stack frame is popped
//    and is free to be overwritten by whatever the caller does next
//    (loop bodies, further calls, usleep()'s own frame, ...). By the time
//    the watchdog's signal arrives later and the handler calls
//    siglongjmp(), it restores the stack pointer to that now-stale,
//    likely-overwritten frame -- a "jump into a function that already
//    returned" bug, undefined behavior, and just as dangerous as the
//    cross-thread longjmp it was meant to replace. The C standard
//    reflects this by restricting where a setjmp/sigsetjmp *call* may
//    textually appear (essentially: directly as the controlling
//    expression of if/while/switch in the SAME function you want control
//    to return to) -- wrapping it in a helper function is exactly the
//    pattern the standard forbids, and the reason isn't pedantry, it's
//    this bug.
//
//    Fix: MSP_ARM_FALLBACK_POINT() below is a macro, not a function, so
//    sigsetjmp() is textually inlined into the monitored thread's own
//    function body. That function's frame is the one still active
//    (blocked in its own wait loop) when the signal arrives, so
//    siglongjmp() has a live, valid frame to return to.

#include <signal.h>

#ifndef MSP_SANDBOX_WATCHDOG_H
#define MSP_SANDBOX_WATCHDOG_H

#include <pthread.h>
#include <setjmp.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MSP_CRITICAL_TEMP_C 85.0
#define MSP_MAX_SRAM_ALLOC (512u * 1024u * 1024u)  // 512MB, per spec Sec. 1
#define MSP_WATCHDOG_POLL_INTERVAL_US 10000        // 10ms target poll interval
#define MSP_FALLBACK_SIGNAL SIGUSR1

// Telemetry snapshot. In production these fields are populated from real
// hardware (thermal zone sysfs, NPU allocator, Titan-style secure element);
// the spec hardcoded them as constants, which meant the watchdog could
// never actually detect a real violation. This struct plus a pluggable
// reader function (msp_telemetry_reader_fn) replace that.
typedef struct {
    float temperature_c;
    size_t sram_usage_bytes;
    bool security_violation;
} msp_telemetry_t;

typedef void (*msp_telemetry_reader_fn)(msp_telemetry_t* out, void* user_data);

// Isolated memory region allocator.
//
// NOTE ON NAMING: the spec called this a "Ring 3 Security Sandbox." Ring 3
// is x86/ARM CPU privilege-level terminology for unprivileged (user-mode)
// execution -- it is not something a userspace mmap() call grants or
// controls; ordinary user-mode code already runs at Ring 3. This function
// does provide a real, useful property (a PROT_EXEC-free memory region, so
// data loaded into it cannot be directly executed as code), but that is
// NOT process isolation or a sandbox on its own. Untrusted adapter code
// still needs to run in a separate process with a restricted syscall
// filter (seccomp-bpf) or a memory-safe interpreter (e.g. a WASM runtime)
// to be meaningfully sandboxed. See docs/SECURITY.md for the full
// threat-model discussion.
void* msp_allocate_isolated_region(size_t size);

// Thread-local rollback context. Exposed here (not hidden as a file-local
// static) because MSP_ARM_FALLBACK_POINT() below must expand directly
// into the monitored thread's own function body -- see fix #4 above for
// why it cannot be reached only through a function call.
extern __thread sigjmp_buf msp_fallback_env;
extern __thread volatile sig_atomic_t msp_fallback_armed;

// Installs the SIGUSR1 handler for the *calling* thread's process-wide
// signal disposition and marks this thread ready to receive the fallback
// signal. This part is safe to put in a real function: it has no
// stack-frame-lifetime requirement, it just sets up state and returns
// normally. Returns 0 on success, -1 on failure (check errno).
int msp_install_fallback_handler(void);

// Arms the fallback rollback point in the CALLING thread. This must be
// used exactly like setjmp/sigsetjmp: as the entire controlling
// expression of an if/while, in the function you want control to return
// to. Do NOT wrap this in another function call -- see fix #4 above.
//
// Usage:
//     if (MSP_ARM_FALLBACK_POINT() != 0) {
//         // reached via the watchdog's fallback signal; roll back here
//     } else {
//         // normal path; do adapter inference work
//     }
#define MSP_ARM_FALLBACK_POINT() \
    (msp_install_fallback_handler(), sigsetjmp(msp_fallback_env, 1))

// Watchdog thread entry point. `arg` must be a pointer to a
// msp_watchdog_config_t (see below). Polls telemetry at
// MSP_WATCHDOG_POLL_INTERVAL_US and signals the monitored thread if a
// violation is detected.
void* msp_telemetry_watchdog_daemon(void* arg);

typedef struct {
    pthread_t monitored_thread;         // thread to signal on violation
    msp_telemetry_reader_fn read_telemetry;
    void* reader_user_data;
    volatile bool* keep_running;        // watchdog loop exits when *keep_running == false
    volatile bool* violation_flag;      // set true right before signaling, for observability/tests
} msp_watchdog_config_t;

#ifdef __cplusplus
}
#endif

#endif  // MSP_SANDBOX_WATCHDOG_H

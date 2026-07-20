// telemetry_server.c
// See telemetry_server.h for the protocol and design rationale.

#include "telemetry_server.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

_Static_assert(MSP_TELEMETRY_SERVER_SOCKET_PATH_MAX == sizeof(((struct sockaddr_un*)0)->sun_path),
               "MSP_TELEMETRY_SERVER_SOCKET_PATH_MAX must match sockaddr_un.sun_path's real size");

// How often the accept loop wakes up to re-check *keep_running. Same
// order of magnitude as MSP_WATCHDOG_POLL_INTERVAL_US's polling
// philosophy (sandbox_watchdog.h): a target responsiveness, not a
// real-time guarantee -- see that header for why Linux can't offer one.
#define MSP_TELEMETRY_SERVER_POLL_INTERVAL_US 200000  // 200ms

static int write_all(int fd, const char* buf, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, buf + written, len - written);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;  // e.g. EPIPE if the client already disconnected;
                        // SIGPIPE itself is expected to be ignored by the
                        // caller (see watchdogd_main.c), so this is a
                        // normal, recoverable failure to report, not a
                        // reason to kill the whole server.
        }
        written += (size_t)n;
    }
    return 0;
}

void* msp_telemetry_server_run(void* arg) {
    msp_telemetry_server_config_t* cfg = (msp_telemetry_server_config_t*)arg;

    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "[TELEMETRY] socket() failed: %s\n", strerror(errno));
        return NULL;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, cfg->socket_path, sizeof(addr.sun_path));
    // Guarantee NUL-termination even if socket_path filled the buffer
    // exactly (memcpy above does not itself guarantee that).
    addr.sun_path[sizeof(addr.sun_path) - 1] = '\0';

    // Remove a stale socket file from a previous unclean shutdown --
    // bind() fails with EADDRINUSE otherwise even though nothing is
    // actually listening anymore. Failure here (e.g. ENOENT because
    // there was nothing to remove) is expected and fine; only bind()'s
    // own result below is checked.
    unlink(addr.sun_path);

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "[TELEMETRY] bind('%s') failed: %s\n", addr.sun_path, strerror(errno));
        close(listen_fd);
        return NULL;
    }
    if (listen(listen_fd, 16) != 0) {
        fprintf(stderr, "[TELEMETRY] listen() failed: %s\n", strerror(errno));
        close(listen_fd);
        unlink(addr.sun_path);
        return NULL;
    }

    while (*cfg->keep_running) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(listen_fd, &readfds);
        struct timeval tv = {
            .tv_sec = 0,
            .tv_usec = MSP_TELEMETRY_SERVER_POLL_INTERVAL_US,
        };
        int ready = select(listen_fd + 1, &readfds, NULL, NULL, &tv);
        if (ready <= 0) {
            // Timeout (ready == 0): loop back around to re-check
            // *keep_running. EINTR (ready < 0): same -- if this thread
            // was interrupted for a fallback signal, the signal handler
            // already transferred control via siglongjmp before select()
            // even returned, so reaching here at all means it was some
            // other, ignorable interruption.
            continue;
        }

        int conn_fd = accept(listen_fd, NULL, NULL);
        if (conn_fd < 0) {
            continue;  // e.g. the connecting client already gave up
        }

        msp_telemetry_t telemetry;
        cfg->read_telemetry(&telemetry, cfg->reader_user_data);

        char line[128];
        int n = snprintf(line, sizeof(line), "%.3f %zu %d\n",
                          (double)telemetry.temperature_c, telemetry.sram_usage_bytes,
                          telemetry.security_violation ? 1 : 0);
        if (n > 0) {
            write_all(conn_fd, line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
        }
        close(conn_fd);
    }

    close(listen_fd);
    unlink(addr.sun_path);
    return NULL;
}

// Standalone test/harness for telemetry_server.c.
//
// Tests telemetry_server.c directly against a scripted reader (not
// thermal_reader.c) to keep this a unit test of the socket
// protocol/server behavior in isolation -- thermal_reader.c already has
// its own dedicated test_thermal_reader.c, and the end-to-end wiring of
// the two together (plus watchdogd_main.c's re-arm loop) is covered by
// the Python integration test in tests/python/test_thermal.py
// (test_watchdogd_integration_*), which exercises the real compiled
// `watchdogd` binary against a simulated sysfs tree.

#include "telemetry_server.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
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
    float temperature_c;
    size_t sram_usage_bytes;
    bool security_violation;
} scripted_reader_state_t;

static void scripted_reader(msp_telemetry_t* out, void* user_data) {
    scripted_reader_state_t* state = (scripted_reader_state_t*)user_data;
    atomic_fetch_add(&state->read_count, 1);
    out->temperature_c = state->temperature_c;
    out->sram_usage_bytes = state->sram_usage_bytes;
    out->security_violation = state->security_violation;
}

// Connects to `socket_path`, reads until EOF (the server closes the
// connection after writing one line -- see telemetry_server.h's
// protocol), and parses the three fields. Returns 0 on success.
static int query_once(const char* socket_path, float* out_temp_c,
                       size_t* out_sram_bytes, int* out_violation) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    // The server may not have bound its listening socket yet if this is
    // called too soon after starting the server thread -- retry briefly
    // rather than requiring the test to guess a fixed startup delay.
    int connected = -1;
    for (int attempt = 0; attempt < 200; attempt++) {
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            connected = 0;
            break;
        }
        usleep(5000);
    }
    if (connected != 0) {
        close(fd);
        return -1;
    }

    char buf[256];
    size_t total = 0;
    for (;;) {
        ssize_t n = read(fd, buf + total, sizeof(buf) - 1 - total);
        if (n < 0) {
            close(fd);
            return -1;
        }
        if (n == 0) {
            break;  // server closed the connection: end of this reading
        }
        total += (size_t)n;
        if (total >= sizeof(buf) - 1) {
            break;
        }
    }
    close(fd);
    buf[total] = '\0';

    double temp;
    unsigned long sram;
    int violation;
    if (sscanf(buf, "%lf %lu %d", &temp, &sram, &violation) != 3) {
        return -1;
    }
    *out_temp_c = (float)temp;
    *out_sram_bytes = (size_t)sram;
    *out_violation = violation;
    return 0;
}

static int run_serves_fresh_reading_per_connection(void) {
    const char* socket_path = "/tmp/msp_test_telemetry_server_basic.sock";

    scripted_reader_state_t reader_state = {0};
    reader_state.temperature_c = 42.5f;
    reader_state.sram_usage_bytes = 123456;
    reader_state.security_violation = false;

    volatile bool keep_running = true;
    msp_telemetry_server_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.socket_path, socket_path, sizeof(cfg.socket_path) - 1);
    cfg.read_telemetry = scripted_reader;
    cfg.reader_user_data = &reader_state;
    cfg.keep_running = &keep_running;

    pthread_t server_thread;
    CHECK(pthread_create(&server_thread, NULL, msp_telemetry_server_run, &cfg) == 0);

    float temp_c;
    size_t sram_bytes;
    int violation;
    CHECK(query_once(socket_path, &temp_c, &sram_bytes, &violation) == 0);
    CHECK(fabsf(temp_c - 42.5f) < 1e-3f);
    CHECK(sram_bytes == 123456);
    CHECK(violation == 0);
    CHECK(atomic_load(&reader_state.read_count) == 1);

    // A second connection gets a second, independently-taken reading
    // (read_count increments again), not a cached copy of the first --
    // this is the property that makes the socket a real live telemetry
    // source rather than a stale snapshot.
    reader_state.temperature_c = 91.0f;
    reader_state.security_violation = true;
    CHECK(query_once(socket_path, &temp_c, &sram_bytes, &violation) == 0);
    CHECK(fabsf(temp_c - 91.0f) < 1e-3f);
    CHECK(violation == 1);
    CHECK(atomic_load(&reader_state.read_count) == 2);

    keep_running = false;
    pthread_join(server_thread, NULL);
    return 0;
}

static int run_shuts_down_cleanly_with_no_connections(void) {
    const char* socket_path = "/tmp/msp_test_telemetry_server_idle_shutdown.sock";

    scripted_reader_state_t reader_state = {0};
    reader_state.temperature_c = 25.0f;

    volatile bool keep_running = true;
    msp_telemetry_server_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.socket_path, socket_path, sizeof(cfg.socket_path) - 1);
    cfg.read_telemetry = scripted_reader;
    cfg.reader_user_data = &reader_state;
    cfg.keep_running = &keep_running;

    pthread_t server_thread;
    CHECK(pthread_create(&server_thread, NULL, msp_telemetry_server_run, &cfg) == 0);

    usleep(50000);  // let it bind and enter its accept loop
    keep_running = false;

    // pthread_join blocks forever if the server's accept-loop-with-timeout
    // logic is broken (e.g. reverted to a plain blocking accept()) -- this
    // is the regression this subtest guards against; ctest's own overall
    // timeout is the real backstop but this keeps the failure mode a
    // logged CHECK failure' worth of signal, not just "the whole test hung".
    CHECK(pthread_join(server_thread, NULL) == 0);

    // The socket file must be cleaned up on graceful shutdown, not left
    // behind to cause a spurious EADDRINUSE on the next run.
    CHECK(access(socket_path, F_OK) != 0);
    return 0;
}

static int run_removes_stale_socket_file_from_previous_run(void) {
    const char* socket_path = "/tmp/msp_test_telemetry_server_stale.sock";

    // Simulate a leftover socket file from an unclean previous shutdown:
    // bind() would fail with EADDRINUSE against this otherwise.
    int stale_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(stale_fd >= 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    unlink(socket_path);
    CHECK(bind(stale_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(stale_fd);  // leaves the socket file on disk, unbound now

    scripted_reader_state_t reader_state = {0};
    reader_state.temperature_c = 30.0f;

    volatile bool keep_running = true;
    msp_telemetry_server_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.socket_path, socket_path, sizeof(cfg.socket_path) - 1);
    cfg.read_telemetry = scripted_reader;
    cfg.reader_user_data = &reader_state;
    cfg.keep_running = &keep_running;

    pthread_t server_thread;
    CHECK(pthread_create(&server_thread, NULL, msp_telemetry_server_run, &cfg) == 0);

    float temp_c;
    size_t sram_bytes;
    int violation;
    CHECK(query_once(socket_path, &temp_c, &sram_bytes, &violation) == 0);
    CHECK(fabsf(temp_c - 30.0f) < 1e-3f);

    keep_running = false;
    pthread_join(server_thread, NULL);
    return 0;
}

int main(void) {
    if (run_serves_fresh_reading_per_connection() != 0) return 1;
    if (run_shuts_down_cleanly_with_no_connections() != 0) return 1;
    if (run_removes_stale_socket_file_from_previous_run() != 0) return 1;
    printf("All telemetry_server tests passed.\n");
    return 0;
}

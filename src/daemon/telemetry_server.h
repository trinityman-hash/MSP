// telemetry_server.h
//
// Closes the second half of the gap STATUS.md's "What's left to do"
// flagged for thermal_reader.c: it existed, and was proven
// type-compatible with msp_watchdog_config_t.read_telemetry, but nothing
// constructed a real daemon around it, and even once one exists, Python's
// ThermalMonitor and the C watchdog would still be two independent
// readers of /sys/class/thermal with no shared data source -- fine until
// they disagree (different aggregation, a filter typo, clock/read skew),
// at which point "the watchdog and the training loop agree on the
// temperature" becomes an assumption instead of a fact.
//
// This module is that shared data source: a small Unix-domain-socket
// server. Any client (the Python side included, see
// msp.thermal.WatchdogTelemetryReader) connects, and gets back a single
// fresh reading, taken via the exact same msp_telemetry_reader_fn the
// watchdog thread itself uses to decide whether to trigger a fallback --
// not a second independent read of the filesystem.
//
// Protocol: one connection = one reading. On connect, the server takes a
// fresh telemetry sample and writes a single line:
//
//     <temperature_c> <sram_usage_bytes> <security_violation:0|1>\n
//
// then closes the connection. Text, not binary: the payload is three
// numbers, there is no performance case for a binary protocol here, and
// text means the protocol is trivially inspectable with `nc -U <path>` or
// `socat -` while debugging. This is a request/response probe ("what's
// the reading right now"), not a subscription/streaming feed -- matching
// ThermalMonitor's own reader contract of "call me and give me one fresh
// value," which is exactly what WatchdogTelemetryReader.__call__ does on
// the other end of this socket.

#ifndef MSP_TELEMETRY_SERVER_H
#define MSP_TELEMETRY_SERVER_H

#include <stdbool.h>

#include "sandbox_watchdog.h"  // for msp_telemetry_reader_fn

#ifdef __cplusplus
extern "C" {
#endif

// Matches sizeof(((struct sockaddr_un*)0)->sun_path) on Linux (108). Not
// derived directly from <sys/un.h> here so this header stays includable
// without pulling in socket headers; telemetry_server.c has a
// compile-time assertion that this still matches the real struct.
#define MSP_TELEMETRY_SERVER_SOCKET_PATH_MAX 108

typedef struct {
    char socket_path[MSP_TELEMETRY_SERVER_SOCKET_PATH_MAX];
    msp_telemetry_reader_fn read_telemetry;
    void* reader_user_data;
    // Server loop exits (unbinding and removing the socket file) once
    // *keep_running == false. Same convention as
    // msp_watchdog_config_t.keep_running.
    volatile bool* keep_running;
} msp_telemetry_server_config_t;

// pthread_create-compatible entry point (arg must point to a
// msp_telemetry_server_config_t, kept alive for the whole call). Binds
// and listens on config->socket_path (removing a stale socket file left
// over from an unclean previous shutdown, if any), then serves one fresh
// reading per accepted connection until *config->keep_running is false or
// an unrecoverable socket error occurs, at which point it unbinds,
// removes the socket file, and returns NULL.
//
// Uses a poll-with-timeout accept loop (not a blocking accept()) so it
// notices *keep_running promptly instead of blocking forever on the next
// connection -- the same tradeoff msp_telemetry_watchdog_daemon makes
// with its own poll interval, for the same reason. It also means this
// call can be interrupted by MSP_FALLBACK_SIGNAL like any other blocking
// call in a thread that has armed MSP_ARM_FALLBACK_POINT() -- see
// watchdogd_main.c, which runs this as the "protected work" a violation
// rolls back.
void* msp_telemetry_server_run(void* arg);

#ifdef __cplusplus
}
#endif

#endif  // MSP_TELEMETRY_SERVER_H

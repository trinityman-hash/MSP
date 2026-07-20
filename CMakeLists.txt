cmake_minimum_required(VERSION 3.16)
project(msp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Required so elora_allocator (a static lib) can be linked into
# msp_native.so (a shared object) below -- without this, static libs are
# compiled without -fPIC by default and the link fails with "relocation
# R_X86_64_PC32 ... can not be used when making a shared object".
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MSP_USE_CUDA "Build EloraAllocator against the CUDA runtime (requires NVIDIA toolkit)" OFF)
option(MSP_BUILD_TESTS "Build the standalone C/C++ tests" ON)
option(MSP_BUILD_PYTHON_BINDINGS "Build the pybind11 extension wiring EloraAllocator into Python" ON)

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# elora_allocator (C++)
# ---------------------------------------------------------------------------
add_library(elora_allocator STATIC
    src/cpp/elora_allocator.cpp
)
target_include_directories(elora_allocator PUBLIC src/cpp)

if(MSP_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(elora_allocator PUBLIC USE_CUDA)
    target_link_libraries(elora_allocator PUBLIC CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# Python bindings (pybind11) -- wires EloraAllocator into src/python/msp
# ---------------------------------------------------------------------------
if(MSP_BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
    )
    if(Python3_FOUND AND PYBIND11_CMAKEDIR_RESULT EQUAL 0)
        find_package(pybind11 CONFIG QUIET)
    endif()
    if(pybind11_FOUND)
        pybind11_add_module(msp_native src/cpp/bindings.cpp)
        target_link_libraries(msp_native PRIVATE elora_allocator)
        # Drop the compiled extension directly into src/python/msp/ so
        # `import msp_native` works from a normal PYTHONPATH=src/python
        # run without any extra install step.
        set_target_properties(msp_native PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python/msp"
        )
    else()
        message(STATUS "pybind11 not found -- skipping msp_native extension "
                        "(AdapterManager will fall back to pure-Python byte tracking)")
    endif()
endif()

# ---------------------------------------------------------------------------
# sandbox_watchdog + integrity_check + thermal_reader (C)
# ---------------------------------------------------------------------------
add_library(msp_daemon STATIC
    src/daemon/sandbox_watchdog.c
    src/daemon/integrity_check.c
    src/daemon/thermal_reader.c
    src/daemon/telemetry_server.c
)
target_include_directories(msp_daemon PUBLIC src/daemon)
target_link_libraries(msp_daemon PUBLIC OpenSSL::Crypto Threads::Threads)

# ---------------------------------------------------------------------------
# watchdogd: the standalone daemon entry point wiring thermal_reader.c into
# sandbox_watchdog.c and exposing telemetry to clients (e.g. Python's
# msp.thermal.WatchdogTelemetryReader) over a Unix domain socket. See
# watchdogd_main.c's own header comment for the full picture.
# ---------------------------------------------------------------------------
add_executable(watchdogd src/daemon/watchdogd_main.c)
target_link_libraries(watchdogd PRIVATE msp_daemon)

# ---------------------------------------------------------------------------
# Tests (plain executables, no external test framework required)
# ---------------------------------------------------------------------------
if(MSP_BUILD_TESTS)
    enable_testing()

    add_executable(test_elora_allocator tests/cpp/test_elora_allocator.cpp)
    target_link_libraries(test_elora_allocator PRIVATE elora_allocator)
    add_test(NAME test_elora_allocator COMMAND test_elora_allocator)

    add_executable(test_integrity_check tests/daemon/test_integrity_check.c)
    target_link_libraries(test_integrity_check PRIVATE msp_daemon)
    add_test(NAME test_integrity_check COMMAND test_integrity_check)

    add_executable(test_watchdog tests/daemon/test_watchdog.c)
    target_link_libraries(test_watchdog PRIVATE msp_daemon)
    add_test(NAME test_watchdog COMMAND test_watchdog)

    add_executable(test_thermal_reader tests/daemon/test_thermal_reader.c)
    target_link_libraries(test_thermal_reader PRIVATE msp_daemon)
    add_test(NAME test_thermal_reader COMMAND test_thermal_reader)

    add_executable(test_telemetry_server tests/daemon/test_telemetry_server.c)
    target_link_libraries(test_telemetry_server PRIVATE msp_daemon)
    add_test(NAME test_telemetry_server COMMAND test_telemetry_server)
endif()

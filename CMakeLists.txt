cmake_minimum_required(VERSION 3.16)
project(msp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(MSP_USE_CUDA "Build EloraAllocator against the CUDA runtime (requires NVIDIA toolkit)" OFF)
option(MSP_BUILD_TESTS "Build the standalone C/C++ tests" ON)

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
# sandbox_watchdog + integrity_check (C)
# ---------------------------------------------------------------------------
add_library(msp_daemon STATIC
    src/daemon/sandbox_watchdog.c
    src/daemon/integrity_check.c
)
target_include_directories(msp_daemon PUBLIC src/daemon)
target_link_libraries(msp_daemon PUBLIC OpenSSL::Crypto Threads::Threads)

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
endif()

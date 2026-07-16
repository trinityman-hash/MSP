// Minimal standalone test for EloraAllocator. Deliberately framework-free
// (no gtest dependency) so it builds anywhere a C++17 compiler exists.
// Returns 0 on success, nonzero (and prints which check failed) otherwise.

#include "elora_allocator.hpp"

#include <cassert>
#include <cstdio>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAILED: %s (line %d)\n", #cond, __LINE__); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    using msp::EloraAllocator;

    // Basic allocate + free-on-eviction.
    {
        EloraAllocator alloc;
        void* p1 = alloc.allocate_kv_context(/*adapter_id=*/1, 4096);
        CHECK(p1 != nullptr);
        CHECK(alloc.bytes_resident_for(1) == 4096);
        CHECK(alloc.resident_adapter_count() == 1);

        alloc.execute_dependency_eviction(1);
        CHECK(alloc.bytes_resident_for(1) == 0);
        CHECK(alloc.resident_adapter_count() == 0);
    }

    // Multiple blocks for the same adapter are all freed together.
    {
        EloraAllocator alloc;
        void* p1 = alloc.allocate_kv_context(2, 1024);
        void* p2 = alloc.allocate_kv_context(2, 2048);
        CHECK(p1 != nullptr && p2 != nullptr);
        CHECK(alloc.bytes_resident_for(2) == 1024 + 2048);

        alloc.execute_dependency_eviction(2);
        CHECK(alloc.bytes_resident_for(2) == 0);
    }

    // Evicting one adapter does not disturb another's blocks
    // (this is the "Orphaned Context" bug the spec was trying to fix --
    // verify adapters are actually isolated from each other).
    {
        EloraAllocator alloc;
        alloc.allocate_kv_context(10, 512);
        alloc.allocate_kv_context(20, 512);
        CHECK(alloc.resident_adapter_count() == 2);

        alloc.execute_dependency_eviction(10);
        CHECK(alloc.resident_adapter_count() == 1);
        CHECK(alloc.bytes_resident_for(20) == 512);
    }

    // Evicting an adapter with no allocations is a safe no-op.
    {
        EloraAllocator alloc;
        alloc.execute_dependency_eviction(999);  // must not crash
        CHECK(alloc.resident_adapter_count() == 0);
    }

    // Zero-size allocation request returns nullptr rather than a bogus
    // pointer, and registers nothing.
    {
        EloraAllocator alloc;
        void* p = alloc.allocate_kv_context(1, 0);
        CHECK(p == nullptr);
        CHECK(alloc.resident_adapter_count() == 0);
    }

    // Destructor cleans up anything still resident (no leak, no crash),
    // exercised implicitly as `alloc` goes out of scope with live blocks.
    {
        EloraAllocator alloc;
        alloc.allocate_kv_context(5, 128);
        // intentionally not evicted -- destructor must free it
    }

    std::printf("All EloraAllocator tests passed.\n");
    return 0;
}

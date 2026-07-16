// elora_allocator.hpp
//
// Dependency-aware KV-cache allocator: binds allocated cache blocks to the
// LoRA adapter that owns them, so unloading an adapter frees every block
// it depends on in one call (Subsystem C in the original spec).
//
// Fix relative to the v3.0 spec:
//   The spec's version called cudaMallocManaged() and cudaFree()
//   unconditionally, which means the file does not even compile on a
//   machine without the CUDA toolkit -- a hard blocker for local
//   development, CI, and any non-NVIDIA target device (which is most of
//   the "edge" devices this project is actually for: phones, laptops).
//   This version keeps the exact same public interface and dependency-tree
//   semantics, but the underlying allocation strategy is selected at
//   compile time via USE_CUDA, defaulting to plain host allocation so the
//   class is usable and testable everywhere. This is a portability fix,
//   not a behavior change: with USE_CUDA defined, allocation still goes
//   through cudaMallocManaged/cudaFree exactly as specified.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace msp {

class EloraAllocator {
public:
    EloraAllocator() = default;
    ~EloraAllocator();

    // Non-copyable: this class owns raw allocations behind the pointers it
    // hands out, and a naive copy would double-free them.
    EloraAllocator(const EloraAllocator&) = delete;
    EloraAllocator& operator=(const EloraAllocator&) = delete;

    // Allocate a KV-cache block of `size` bytes bound to `adapter_id`.
    // Returns nullptr on allocation failure (caller must check).
    void* allocate_kv_context(int adapter_id, std::size_t size);

    // Free every block bound to `adapter_id`. Safe to call on an
    // adapter_id with no allocations (no-op).
    void execute_dependency_eviction(int adapter_id);

    // Introspection, mainly for tests: total bytes currently attributed to
    // an adapter_id, and how many distinct adapters are resident.
    std::size_t bytes_resident_for(int adapter_id) const;
    std::size_t resident_adapter_count() const;

private:
    struct Block {
        void* ptr;
        std::size_t size;
    };

    std::unordered_map<int, std::vector<Block>> dependency_tree_;

    void* raw_allocate(std::size_t size);
    void raw_free(void* ptr);
};

}  // namespace msp

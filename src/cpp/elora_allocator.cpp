// elora_allocator.cpp
// See elora_allocator.hpp for the rationale behind the USE_CUDA switch.

#include "elora_allocator.hpp"

#include <cstdlib>
#include <iostream>

namespace msp {

void* EloraAllocator::raw_allocate(std::size_t size) {
#ifdef USE_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
#else
    // Plain host allocation. Matches the spec's "unified/managed memory"
    // semantics in spirit (single pointer valid for the allocator's own
    // bookkeeping); the CUDA build provides the real unified-memory
    // behavior for GPU access.
    return std::malloc(size);
#endif
}

void EloraAllocator::raw_free(void* ptr) {
#ifdef USE_CUDA
    cudaFree(ptr);
#else
    std::free(ptr);
#endif
}

EloraAllocator::~EloraAllocator() {
    // Defensive cleanup: free anything still resident at destruction time
    // rather than leaking it, even if a caller forgot to evict.
    for (auto& entry : dependency_tree_) {
        for (auto& block : entry.second) {
            raw_free(block.ptr);
        }
    }
    dependency_tree_.clear();
}

void* EloraAllocator::allocate_kv_context(int adapter_id, std::size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = raw_allocate(size);
    if (ptr == nullptr) {
        return nullptr;
    }
    dependency_tree_[adapter_id].push_back(Block{ptr, size});
    return ptr;
}

void EloraAllocator::execute_dependency_eviction(int adapter_id) {
    auto it = dependency_tree_.find(adapter_id);
    if (it == dependency_tree_.end()) {
        return;
    }
    for (auto& block : it->second) {
        raw_free(block.ptr);
    }
    dependency_tree_.erase(it);
    std::cout << "[ELORA] Orphaned Context Cleared for Adapter: " << adapter_id << "\n";
}

std::size_t EloraAllocator::bytes_resident_for(int adapter_id) const {
    auto it = dependency_tree_.find(adapter_id);
    if (it == dependency_tree_.end()) {
        return 0;
    }
    std::size_t total = 0;
    for (const auto& block : it->second) {
        total += block.size;
    }
    return total;
}

std::size_t EloraAllocator::resident_adapter_count() const {
    return dependency_tree_.size();
}

}  // namespace msp

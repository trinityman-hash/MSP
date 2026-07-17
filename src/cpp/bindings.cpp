// bindings.cpp
//
// Exposes EloraAllocator (src/cpp/elora_allocator.*) to Python as
// `msp.msp_native.EloraAllocator` (built into the `msp` package directory
// so it's imported with `from . import msp_native`), so AdapterManager can
// perform real KV-cache allocation/eviction instead of only tracking byte
// counts.
//
// This directly addresses the #1 item in docs/STATUS.md's "What's left
// to do" list: the Python, C++, and C subsystems previously existed as
// three independent, tested-but-unconnected islands. This is the first
// real bridge between two of them.
//
// Build: this module is optional. AdapterManager (Python) detects at
// import time whether `msp_native` is importable and falls back to
// pure-Python byte-count tracking if it isn't (e.g. on a machine without
// a C++ toolchain, or before this extension has been built) -- see
// adapter_manager.py's `_native` import guard. Nothing breaks if this
// extension is absent; you just don't get real memory allocation behind
// the budget accounting.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "elora_allocator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(msp_native, m) {
    m.doc() = "Native (C++) bindings for MSP's EloraAllocator";

    py::class_<msp::EloraAllocator>(m, "EloraAllocator")
        .def(py::init<>())
        .def(
            "allocate_kv_context",
            [](msp::EloraAllocator& self, int adapter_id, std::size_t size) -> py::object {
                void* ptr = self.allocate_kv_context(adapter_id, size);
                if (ptr == nullptr) {
                    return py::none();
                }
                // Expose the pointer as an integer address. Python callers
                // in this project treat this as an opaque handle (they
                // never dereference it directly); it's only used to prove
                // a real allocation happened and to detect nullptr
                // (allocation failure / zero-size request).
                return py::cast(reinterpret_cast<std::uintptr_t>(ptr));
            },
            py::arg("adapter_id"), py::arg("size"),
            "Allocate a KV-cache block bound to adapter_id. Returns an "
            "opaque integer handle, or None on failure/zero-size request.")
        .def("execute_dependency_eviction", &msp::EloraAllocator::execute_dependency_eviction,
             py::arg("adapter_id"),
             "Free every block bound to adapter_id (no-op if none exist).")
        .def("bytes_resident_for", &msp::EloraAllocator::bytes_resident_for,
             py::arg("adapter_id"))
        .def("resident_adapter_count", &msp::EloraAllocator::resident_adapter_count);
}

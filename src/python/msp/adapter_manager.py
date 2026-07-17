"""
AdapterManager: tracks which "structural plugin" adapters are resident,
enforces the spec's 512MB on-device memory boundary, and mirrors the
C++ EloraAllocator's dependency-tree eviction (unloading an adapter frees
everything it owns) at the Python/PyTorch level.

The original spec enforced no memory budget at the Python layer at all --
the 512MB limit only appeared as a `#define MAX_SRAM_ALLOC` guard deep in
the C sandbox allocator. That means a caller could register far more
adapters than fit in the budget and only find out when a *different*
subsystem (the C allocator) started returning NULL. This manager fails
fast, in Python, at registration time, with a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .plugin_layer import StructuralPluginLayer

DEFAULT_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024  # 512MB, per spec Sec. 1

# Optional native binding (src/cpp/bindings.cpp -> msp_native), built via
# `cmake -B build && cmake --build build` with pybind11 available. This is
# the fix for STATUS.md item #1 ("wire the subsystems together"):
# previously AdapterManager only tracked byte counts in pure Python and
# never actually touched EloraAllocator. When the extension is present,
# every load/unload now performs a real allocation/eviction through it;
# when it isn't (no C++ toolchain, or simply not built yet), everything
# falls back to the pure-Python byte-tracking behavior this class always
# had, so nothing that worked before is broken by this becoming optional.
try:
    from . import msp_native  # type: ignore
    _NATIVE_AVAILABLE = True
except ImportError:
    msp_native = None  # type: ignore
    _NATIVE_AVAILABLE = False


class AdapterBudgetError(RuntimeError):
    """Raised when loading an adapter would exceed the memory budget."""


@dataclass
class _AdapterRecord:
    adapter_id: str
    layers: Dict[str, StructuralPluginLayer] = field(default_factory=dict)

    @property
    def byte_size(self) -> int:
        return sum(layer.parameter_bytes() for layer in self.layers.values())


class AdapterManager:
    """
    Registry of resident adapters, keyed by adapter_id. An "adapter" can
    span multiple StructuralPluginLayer instances (e.g. one per attention
    block) that are loaded and evicted together, mirroring the C++
    dependency tree in elora_allocator.cpp.
    """

    def __init__(self, memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES) -> None:
        if memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        self.memory_budget_bytes = memory_budget_bytes
        self._adapters: Dict[str, _AdapterRecord] = {}
        self._active_adapter_id: Optional[str] = None

        # Native backing allocator (see module-level _NATIVE_AVAILABLE).
        # EloraAllocator's C++ interface is keyed by int, but adapter_id is
        # a string here (matching the spec's usage), so we maintain a
        # stable string->int mapping alongside it.
        self._native = msp_native.EloraAllocator() if _NATIVE_AVAILABLE else None
        self._native_ids: Dict[str, int] = {}
        self._next_native_id = 1

    @property
    def native_backed(self) -> bool:
        """True if this instance is actually allocating real memory
        through EloraAllocator (msp_native extension built and importable),
        rather than only tracking byte counts in Python."""
        return self._native is not None

    # ------------------------------------------------------------------

    @property
    def resident_bytes(self) -> int:
        return sum(record.byte_size for record in self._adapters.values())

    @property
    def active_adapter_id(self) -> Optional[str]:
        return self._active_adapter_id

    def resident_adapter_ids(self) -> list[str]:
        return list(self._adapters.keys())

    # ------------------------------------------------------------------

    def load_adapter(self, adapter_id: str, layers: Dict[str, StructuralPluginLayer]) -> None:
        """
        Register a new adapter's layers. Raises AdapterBudgetError if doing
        so would exceed the configured memory budget -- checked *before*
        committing, so a rejected load never leaves partial state behind.
        """
        if adapter_id in self._adapters:
            raise ValueError(f"adapter_id '{adapter_id}' is already loaded; unload it first")

        incoming_bytes = sum(layer.parameter_bytes() for layer in layers.values())
        projected_total = self.resident_bytes + incoming_bytes
        if projected_total > self.memory_budget_bytes:
            raise AdapterBudgetError(
                f"loading adapter '{adapter_id}' ({incoming_bytes} bytes) would bring "
                f"resident total to {projected_total} bytes, exceeding the "
                f"{self.memory_budget_bytes}-byte budget"
            )

        self._adapters[adapter_id] = _AdapterRecord(adapter_id=adapter_id, layers=dict(layers))

        if self._native is not None:
            native_id = self._next_native_id
            self._next_native_id += 1
            self._native_ids[adapter_id] = native_id
            for layer in layers.values():
                nbytes = layer.parameter_bytes()
                if nbytes == 0:
                    continue
                handle = self._native.allocate_kv_context(native_id, nbytes)
                if handle is None:
                    # Real allocation failed (e.g. host out of memory) even
                    # though our own budget accounting said this should
                    # fit. Roll back everything for this adapter -- both
                    # the Python-side record and any native blocks already
                    # reserved for it -- so we don't leave partial state.
                    self._native.execute_dependency_eviction(native_id)
                    del self._native_ids[adapter_id]
                    del self._adapters[adapter_id]
                    raise AdapterBudgetError(
                        f"adapter '{adapter_id}' passed the {self.memory_budget_bytes}-byte "
                        f"budget check but the underlying host allocation for a "
                        f"{nbytes}-byte block failed (out of memory?)"
                    )

    def unload_adapter(self, adapter_id: str) -> None:
        """
        Evict an adapter and everything it owns -- the Python-level
        equivalent of EloraAllocator::execute_dependency_eviction. Also
        detaches gradient-gate hooks so no stale hooks outlive the layer.
        """
        record = self._adapters.pop(adapter_id, None)
        if record is None:
            return
        for layer in record.layers.values():
            layer.clear_gradient_gate()
        if self._active_adapter_id == adapter_id:
            self._active_adapter_id = None

        if self._native is not None:
            native_id = self._native_ids.pop(adapter_id, None)
            if native_id is not None:
                self._native.execute_dependency_eviction(native_id)

    def activate(self, adapter_id: str) -> None:
        if adapter_id not in self._adapters:
            raise KeyError(f"adapter_id '{adapter_id}' is not loaded")
        self._active_adapter_id = adapter_id

    def get_layer(self, adapter_id: str, layer_name: str) -> StructuralPluginLayer:
        return self._adapters[adapter_id].layers[layer_name]

    def swap(self, from_adapter_id: Optional[str], to_adapter_id: str) -> None:
        """
        Convenience helper for the "hot-swap" workflow: activate a
        different already-loaded adapter. Does NOT unload `from_adapter_id`
        -- callers that want the old adapter evicted (freeing memory)
        should call unload_adapter explicitly, since keeping several
        adapters resident and merely switching which is active is a valid,
        common use case too.
        """
        self.activate(to_adapter_id)

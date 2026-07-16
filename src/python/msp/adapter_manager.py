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

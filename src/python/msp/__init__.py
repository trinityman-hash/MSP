"""
MSP (Modular Structural Plugins) — reference Python implementation.

This package provides a CPU/GPU-portable reference implementation of
hot-swappable, LoRA-style "structural plugins" for on-device model adaptation.

See docs/ARCHITECTURE.md in the repository root for the design rationale and
a list of the correctness issues found (and fixed) relative to the original
v3.0 specification.
"""

from .plugin_layer import StructuralPluginLayer
from .adapter_manager import AdapterManager, AdapterBudgetError
from .thermal import ThermalMonitor, ThermalReading

__all__ = [
    "StructuralPluginLayer",
    "AdapterManager",
    "AdapterBudgetError",
    "ThermalMonitor",
    "ThermalReading",
]

__version__ = "0.1.0"

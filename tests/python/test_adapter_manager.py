import pytest
import torch.nn as nn

from msp.adapter_manager import AdapterManager, AdapterBudgetError
from msp.plugin_layer import StructuralPluginLayer

_native_probe = AdapterManager()
_NATIVE_BUILT = _native_probe.native_backed
_SKIP_REASON = (
    "msp_native extension not built -- run `cmake -B build && "
    "cmake --build build` first (see CMakeLists.txt's "
    "MSP_BUILD_PYTHON_BINDINGS option)"
)
requires_native = pytest.mark.skipif(not _NATIVE_BUILT, reason=_SKIP_REASON)


def _adapter(in_f=1024, out_f=1024, rank=8):
    base = nn.Linear(in_f, out_f)
    return {"layer0": StructuralPluginLayer(base, rank=rank)}


def test_load_and_activate():
    mgr = AdapterManager()
    mgr.load_adapter("a1", _adapter())
    mgr.activate("a1")
    assert mgr.active_adapter_id == "a1"
    assert "a1" in mgr.resident_adapter_ids()


def test_duplicate_load_raises():
    mgr = AdapterManager()
    mgr.load_adapter("a1", _adapter())
    with pytest.raises(ValueError):
        mgr.load_adapter("a1", _adapter())


def test_unload_frees_budget_for_next_load():
    # Budget sized to fit exactly one adapter of this shape at a time.
    one_adapter = _adapter(in_f=512, out_f=512, rank=8)
    size = sum(l.parameter_bytes() for l in one_adapter.values())
    mgr = AdapterManager(memory_budget_bytes=size)

    mgr.load_adapter("a1", _adapter(in_f=512, out_f=512, rank=8))
    with pytest.raises(AdapterBudgetError):
        mgr.load_adapter("a2", _adapter(in_f=512, out_f=512, rank=8))

    mgr.unload_adapter("a1")
    # Should now succeed since a1's bytes were freed.
    mgr.load_adapter("a2", _adapter(in_f=512, out_f=512, rank=8))
    assert mgr.resident_adapter_ids() == ["a2"]


def test_budget_rejection_leaves_no_partial_state():
    mgr = AdapterManager(memory_budget_bytes=1)  # impossibly small
    with pytest.raises(AdapterBudgetError):
        mgr.load_adapter("a1", _adapter())
    assert mgr.resident_adapter_ids() == []
    assert mgr.resident_bytes == 0


def test_unloading_active_adapter_clears_active_id():
    mgr = AdapterManager()
    mgr.load_adapter("a1", _adapter())
    mgr.activate("a1")
    mgr.unload_adapter("a1")
    assert mgr.active_adapter_id is None


def test_activate_unknown_adapter_raises():
    mgr = AdapterManager()
    with pytest.raises(KeyError):
        mgr.activate("does-not-exist")


def test_unload_unknown_adapter_is_a_noop():
    mgr = AdapterManager()
    mgr.unload_adapter("never-loaded")  # should not raise


def test_native_backend_availability_is_reported_honestly():
    """native_backed reflects whether msp_native was actually importable --
    this test just checks the property doesn't crash and matches the
    module-level probe used to decide which other tests below to run."""
    mgr = AdapterManager()
    assert mgr.native_backed == _NATIVE_BUILT


@requires_native
def test_native_allocation_actually_happens_on_load():
    """Regression test for STATUS.md item #1 ('wire the subsystems
    together'): loading an adapter must reserve real memory through
    EloraAllocator, not just increment a Python-side counter."""
    mgr = AdapterManager()
    layers = _adapter(in_f=512, out_f=512, rank=8)
    expected_bytes = sum(l.parameter_bytes() for l in layers.values())

    mgr.load_adapter("a1", layers)

    native_id = mgr._native_ids["a1"]
    assert mgr._native.bytes_resident_for(native_id) == expected_bytes
    assert mgr._native.resident_adapter_count() == 1


@requires_native
def test_native_eviction_actually_frees_memory_on_unload():
    mgr = AdapterManager()
    mgr.load_adapter("a1", _adapter())
    native_id = mgr._native_ids["a1"]
    assert mgr._native.resident_adapter_count() == 1

    mgr.unload_adapter("a1")

    # The adapter_id -> native_id mapping is cleaned up on the Python side...
    assert "a1" not in mgr._native_ids
    # ...and the underlying native allocator actually freed the block.
    assert mgr._native.bytes_resident_for(native_id) == 0
    assert mgr._native.resident_adapter_count() == 0


@requires_native
def test_native_ids_are_independent_across_adapters():
    """Two different adapter_ids must map to two different native ids, so
    evicting one never touches the other's real allocation."""
    mgr = AdapterManager()
    mgr.load_adapter("a1", _adapter(in_f=256, out_f=256, rank=4))
    mgr.load_adapter("a2", _adapter(in_f=256, out_f=256, rank=4))

    id1, id2 = mgr._native_ids["a1"], mgr._native_ids["a2"]
    assert id1 != id2
    assert mgr._native.resident_adapter_count() == 2

    mgr.unload_adapter("a1")
    assert mgr._native.bytes_resident_for(id1) == 0
    assert mgr._native.bytes_resident_for(id2) > 0
    assert mgr._native.resident_adapter_count() == 1


@requires_native
def test_budget_rejection_never_touches_native_allocator():
    """The Python-side budget check must reject an over-budget load before
    any native allocation is attempted -- otherwise a rejected load could
    still leak real memory."""
    mgr = AdapterManager(memory_budget_bytes=1)  # impossibly small
    with pytest.raises(AdapterBudgetError):
        mgr.load_adapter("a1", _adapter())
    assert mgr._native.resident_adapter_count() == 0

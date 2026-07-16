import pytest
import torch.nn as nn

from msp.adapter_manager import AdapterManager, AdapterBudgetError
from msp.plugin_layer import StructuralPluginLayer


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

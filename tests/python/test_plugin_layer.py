import asyncio

import pytest
import torch
import torch.nn as nn

from msp.plugin_layer import StructuralPluginLayer
from msp.thermal import ThermalMonitor, ScriptedThermalReader


def _make_layer(in_f=16, out_f=8, rank=4, dtype=torch.float32):
    base = nn.Linear(in_f, out_f, dtype=dtype)
    return StructuralPluginLayer(base, rank=rank, alpha=8.0, dtype=dtype)


def test_forward_shape_and_finiteness():
    layer = _make_layer()
    x = torch.randn(3, 16)
    y = layer(x)
    assert y.shape == (3, 8)
    assert torch.isfinite(y).all()


def test_zero_initialized_B_means_adapter_starts_as_identity_passthrough():
    """plugin_matrix_B is zero-initialized (per spec), so at init the
    adapter must contribute nothing and forward() must equal the base
    layer's own output exactly."""
    base = nn.Linear(16, 8)
    layer = StructuralPluginLayer(base, rank=4, alpha=8.0)
    x = torch.randn(5, 16)
    with torch.no_grad():
        expected = base(x)
        actual = layer(x)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_base_layer_is_frozen():
    layer = _make_layer()
    assert layer.base_layer.weight.requires_grad is False
    assert layer.plugin_matrix_A.requires_grad is True
    assert layer.plugin_matrix_B.requires_grad is True


def test_adapter_dtype_matches_base_layer_by_default():
    """Regression test for spec bug #2: adapter dtype silently forced to
    float32 regardless of base layer dtype."""
    layer = _make_layer(dtype=torch.float64)
    assert layer.plugin_matrix_A.dtype == torch.float64
    assert layer.plugin_matrix_B.dtype == torch.float64


def test_forward_async_matches_sync_forward():
    layer = _make_layer()
    x = torch.randn(2, 16)
    with torch.no_grad():
        sync_out = layer(x)
        async_out = asyncio.run(layer.forward_async(x))
    assert torch.allclose(sync_out, async_out, atol=1e-6)


def test_forward_async_does_not_require_cuda():
    """Regression test for spec bug #1: unconditional torch.cuda.* calls
    crashed on CPU-only machines. This must succeed with no GPU present."""
    layer = _make_layer()
    x = torch.randn(2, 16)
    result = asyncio.run(layer.forward_async(x))
    assert result.shape == (2, 8)


def test_gradient_gate_freezes_expected_row_fraction_when_hot():
    # NOTE: plugin_matrix_B is zero-initialized by design (matches spec),
    # which means dL/d(plugin_matrix_A) is mathematically zero at
    # initialization regardless of gating -- a standard LoRA property, not
    # a bug. So this test (and the one below) checks plugin_matrix_B,
    # which does receive a nonzero gradient at init.
    layer = _make_layer(out_f=10, rank=4)
    monitor = ThermalMonitor(
        reader=ScriptedThermalReader([80.0]),  # hot -> throttling
        freeze_threshold_c=75.0,
        freeze_ratio=0.2,  # keep every 5th row
    )
    layer.gate_gradients(monitor)

    x = torch.randn(4, 16)
    out = layer(x)
    out.sum().backward()

    grad_b = layer.plugin_matrix_B.grad
    assert grad_b is not None
    nonzero_rows = (grad_b.abs().sum(dim=1) > 0).sum().item()
    # out_f=10, stride=5 -> rows 0 and 5 stay active -> exactly 2 nonzero rows
    assert nonzero_rows == 2


def test_gradient_gate_does_not_freeze_when_cool():
    layer = _make_layer(out_f=10, rank=4)
    monitor = ThermalMonitor(
        reader=ScriptedThermalReader([25.0]),  # cool -> no throttling
        freeze_threshold_c=75.0,
        freeze_ratio=0.2,
    )
    layer.gate_gradients(monitor)

    x = torch.randn(4, 16)
    out = layer(x)
    out.sum().backward()

    grad_b = layer.plugin_matrix_B.grad
    nonzero_rows = (grad_b.abs().sum(dim=1) > 0).sum().item()
    assert nonzero_rows == 10  # all rows active


def test_clear_gradient_gate_removes_hooks():
    layer = _make_layer(rank=10)
    monitor = ThermalMonitor(reader=ScriptedThermalReader([80.0]), freeze_ratio=0.2)
    layer.gate_gradients(monitor)
    assert len(layer._grad_gate_handles) == 2
    layer.clear_gradient_gate()
    assert len(layer._grad_gate_handles) == 0


def test_routing_gate_bypasses_adapter_without_freeing_it():
    base = nn.Linear(16, 8)
    layer = StructuralPluginLayer(base, rank=4, alpha=8.0)
    # Force B away from its zero-init so the adapter actually has an effect
    # once the gate is re-enabled -- otherwise this test can't distinguish
    # "gate off" from "B happens to be zero".
    with torch.no_grad():
        layer.plugin_matrix_B.fill_(0.5)

    x = torch.randn(3, 16)
    with torch.no_grad():
        base_only = base(x)
        layer.routing_gate_enabled = False
        gated_off = layer(x)
        layer.routing_gate_enabled = True
        gated_on = layer(x)

    assert torch.allclose(gated_off, base_only, atol=1e-6)
    assert not torch.allclose(gated_on, base_only, atol=1e-6)
    # Disabling the gate must not touch the adapter's own parameters --
    # it's a compute bypass, not an eviction.
    assert torch.all(layer.plugin_matrix_B == 0.5)


def test_rank_must_be_positive():
    base = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        StructuralPluginLayer(base, rank=0)


def test_parameter_bytes_matches_tensor_sizes():
    layer = _make_layer(in_f=16, out_f=8, rank=4, dtype=torch.float32)
    expected = (4 * 16 + 8 * 4) * 4  # elements * 4 bytes (float32)
    assert layer.parameter_bytes() == expected

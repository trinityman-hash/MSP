"""
Tests for examples/e2e_training.py.

examples/ is deliberately outside the installed `msp` package (it is a
demo, not library code), so it is not importable via the normal
`from msp... import ...` path the rest of this test suite uses. It is
loaded here directly from its file path via importlib, rather than
adding examples/ to sys.path, to keep this test file self-contained and
avoid changing import behavior for any other test.

These tests exercise the actual integration gap docs/STATUS.md's item #2
called out: everything in `msp` is unit-tested one piece at a time, but
nothing exercised StructuralPluginLayer, AdapterManager, ThermalMonitor,
and msp.persistence together, wired into a real multi-layer model, across
a real training loop.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "e2e_training.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("e2e_training_example", _EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ex = _load_example_module()


VOCAB, D_MODEL, N_HEADS, FFN_DIM, N_LAYERS = 12, 32, 4, 64, 2
RANK, ALPHA = 8, 16.0


def _build_model(seed=0):
    torch.manual_seed(seed)
    embed, blocks, head = ex.build_frozen_base(VOCAB, D_MODEL, N_HEADS, FFN_DIM, N_LAYERS)
    base_linears = ex.collect_base_sublayers(blocks)
    return embed, blocks, head, base_linears


def test_frozen_base_has_no_trainable_parameters():
    embed, blocks, head, _ = _build_model()
    assert all(not p.requires_grad for p in embed.parameters())
    assert all(not p.requires_grad for p in blocks.parameters())
    assert all(not p.requires_grad for p in head.parameters())


def test_attach_adapters_rejects_already_wrapped_sublayers():
    """collect_base_sublayers must fail loudly, not silently wrap a
    StructuralPluginLayer as if it were the original nn.Linear, if called
    after plug_in() has already swapped a block's sublayers."""
    embed, blocks, head, base_linears = _build_model()
    layers = ex.attach_adapters(base_linears, RANK, ALPHA)
    ex.plug_in(blocks, layers)
    with pytest.raises(TypeError):
        ex.collect_base_sublayers(blocks)


def test_two_adapters_over_same_base_are_independent():
    embed, blocks, head, base_linears = _build_model()
    layers_a = ex.attach_adapters(base_linears, RANK, ALPHA)
    layers_b = ex.attach_adapters(base_linears, RANK, ALPHA)
    for name in layers_a:
        assert layers_a[name] is not layers_b[name]
        assert layers_a[name].base_layer is layers_b[name].base_layer  # same frozen base
        assert layers_a[name].plugin_matrix_A is not layers_b[name].plugin_matrix_A


def test_training_loop_reduces_loss_substantially():
    """A real forward/backward/optimizer.step() loop through the whole
    multi-layer model, on a fixed seed, must make real progress -- not
    just run without crashing."""
    embed, blocks, head, base_linears = _build_model(seed=1)
    layers = ex.attach_adapters(base_linears, RANK, ALPHA)
    ex.plug_in(blocks, layers)

    generator = torch.Generator().manual_seed(42)
    losses = ex.train_adapter(
        embed, blocks, head, layers,
        vocab_size=VOCAB, shift=1, steps=120, generator=generator,
    )

    assert len(losses) == 120
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    initial_avg = sum(losses[:10]) / 10
    final_avg = sum(losses[-10:]) / 10
    # Random-guess cross-entropy for vocab=12 is ln(12) ~= 2.48; a working
    # training loop should land far below that, not just decrease a little.
    assert final_avg < initial_avg * 0.1
    assert final_avg < 0.5


def test_training_only_updates_adapter_parameters():
    """The frozen base (embed, head, and the un-adapted q_proj/k_proj of
    every block) must be bit-for-bit unchanged after training -- a bug
    that accidentally let the optimizer touch base parameters would be
    invisible in the per-layer unit tests, since those never build a
    composed model at all."""
    embed, blocks, head, base_linears = _build_model(seed=2)
    layers = ex.attach_adapters(base_linears, RANK, ALPHA)
    ex.plug_in(blocks, layers)

    before = {name: p.detach().clone() for name, p in embed.named_parameters()}
    before.update({f"head.{name}": p.detach().clone() for name, p in head.named_parameters()})
    for i, blk in enumerate(blocks):
        before[f"block{i}.q_proj.weight"] = blk.q_proj.weight.detach().clone()
        before[f"block{i}.k_proj.weight"] = blk.k_proj.weight.detach().clone()

    ex.train_adapter(embed, blocks, head, layers, vocab_size=VOCAB, shift=1, steps=10)

    for name, p in embed.named_parameters():
        assert torch.equal(before[name], p)
    for name, p in head.named_parameters():
        assert torch.equal(before[f"head.{name}"], p)
    for i, blk in enumerate(blocks):
        assert torch.equal(before[f"block{i}.q_proj.weight"], blk.q_proj.weight)
        assert torch.equal(before[f"block{i}.k_proj.weight"], blk.k_proj.weight)

    # Meanwhile the adapters themselves must actually have moved.
    for layer in layers.values():
        assert not torch.all(layer.plugin_matrix_B == 0.0)


def test_thermal_gating_freezes_adapter_rows_across_the_whole_model_when_hot():
    """Extends the per-layer gradient-gating unit tests in
    test_plugin_layer.py to the composed multi-layer model: every
    adapter's gate must be armed and respected inside a real training
    step, not just a single isolated layer.forward()/backward()."""
    from msp.thermal import ScriptedThermalReader

    embed, blocks, head, base_linears = _build_model(seed=3)
    layers = ex.attach_adapters(base_linears, RANK, ALPHA)
    ex.plug_in(blocks, layers)

    monitor = ex.ThermalMonitor(
        reader=ScriptedThermalReader([80.0]),  # hot -> throttling every step
        freeze_threshold_c=75.0,
        freeze_ratio=0.2,
    )

    ex.train_adapter(
        embed, blocks, head, layers,
        vocab_size=VOCAB, shift=1, steps=1, thermal_monitor=monitor,
    )

    for layer in layers.values():
        out_f = layer.plugin_matrix_B.shape[0]
        stride = max(1, round(1.0 / monitor.freeze_ratio))
        expected_active_rows = len(range(0, out_f, stride))
        nonzero_rows = (layer.plugin_matrix_B.grad.abs().sum(dim=1) > 0).sum().item()
        assert nonzero_rows == expected_active_rows

    # gate_gradients/clear_gradient_gate must not leak hooks past the call.
    for layer in layers.values():
        assert layer._grad_gate_handles == []


def test_adapter_manager_budget_and_hot_swap_are_exercised_for_real():
    """Sizes the manager's budget directly from these adapters' own
    parameter_bytes() (not an arbitrary synthetic number), and confirms
    the load/reject/evict/reload cycle actually happens end-to-end."""
    embed, blocks, head, base_linears = _build_model(seed=4)
    layers_a = ex.attach_adapters(base_linears, RANK, ALPHA)
    layers_b = ex.attach_adapters(base_linears, RANK, ALPHA)

    one_adapter_bytes = sum(layer.parameter_bytes() for layer in layers_a.values())
    manager = ex.AdapterManager(memory_budget_bytes=int(one_adapter_bytes * 1.5))

    manager.load_adapter("a", layers_a)
    with pytest.raises(ex.AdapterBudgetError):
        manager.load_adapter("b", layers_b)

    manager.unload_adapter("a")
    manager.load_adapter("b", layers_b)  # now fits
    assert manager.resident_adapter_ids() == ["b"]
    assert manager.resident_bytes == one_adapter_bytes


def test_persistence_round_trip_preserves_trained_behavior(tmp_path):
    """Train an adapter for real, save it, reconstruct it from disk into
    fresh StructuralPluginLayer instances, and confirm the reloaded
    adapter produces bit-for-bit identical output to the trained one --
    the full "hot-swappable... loaded... evicted independently" promise
    from the README, exercised against a real multi-layer model instead
    of a single layer in isolation."""
    embed, blocks, head, base_linears = _build_model(seed=5)
    layers = ex.attach_adapters(base_linears, RANK, ALPHA)
    ex.plug_in(blocks, layers)
    ex.train_adapter(embed, blocks, head, layers, vocab_size=VOCAB, shift=1, steps=30)

    tokens, _ = ex.synthetic_batch(VOCAB, batch=4, seq_len=4, shift=1)
    with torch.no_grad():
        trained_output = ex.forward(embed, blocks, head, tokens)

    save_path = tmp_path / "adapter.safetensors"
    from msp import save_adapter, load_adapter

    save_adapter(save_path, layers)
    reloaded_layers = load_adapter(save_path, base_layers=base_linears)
    ex.plug_in(blocks, reloaded_layers)

    with torch.no_grad():
        reloaded_output = ex.forward(embed, blocks, head, tokens)

    assert torch.allclose(trained_output, reloaded_output, atol=1e-6)

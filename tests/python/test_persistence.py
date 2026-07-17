import torch
import torch.nn as nn
import pytest

from msp.persistence import save_adapter, load_adapter
from msp.plugin_layer import StructuralPluginLayer


def _trained_layer(in_f=16, out_f=8, rank=4, alpha=8.0):
    """A layer with non-trivial (non-zero-init) A/B weights, so a
    round-trip test can actually distinguish 'loaded correctly' from
    'happened to load zeros, which is also what a fresh layer has'."""
    base = nn.Linear(in_f, out_f)
    layer = StructuralPluginLayer(base, rank=rank, alpha=alpha)
    with torch.no_grad():
        layer.plugin_matrix_A.copy_(torch.randn_like(layer.plugin_matrix_A))
        layer.plugin_matrix_B.copy_(torch.randn_like(layer.plugin_matrix_B))
    return base, layer


def test_save_and_load_round_trip_preserves_weights(tmp_path):
    base, layer = _trained_layer()
    path = tmp_path / "adapter.safetensors"

    save_adapter(path, {"proj": layer})

    fresh_base = nn.Linear(16, 8)
    with torch.no_grad():
        fresh_base.weight.copy_(base.weight)
        fresh_base.bias.copy_(base.bias)

    loaded = load_adapter(path, {"proj": fresh_base})
    loaded_layer = loaded["proj"]

    assert torch.equal(loaded_layer.plugin_matrix_A, layer.plugin_matrix_A)
    assert torch.equal(loaded_layer.plugin_matrix_B, layer.plugin_matrix_B)


def test_save_and_load_round_trip_preserves_metadata(tmp_path):
    base, layer = _trained_layer(rank=6, alpha=12.0)
    layer.routing_gate_enabled = False
    path = tmp_path / "adapter.safetensors"

    save_adapter(path, {"proj": layer})
    loaded = load_adapter(path, {"proj": base})
    loaded_layer = loaded["proj"]

    assert loaded_layer.rank == 6
    assert loaded_layer.scaling == pytest.approx(12.0 / 6)
    assert loaded_layer.routing_gate_enabled is False


def test_loaded_layer_produces_identical_output_to_original(tmp_path):
    base, layer = _trained_layer()
    path = tmp_path / "adapter.safetensors"
    save_adapter(path, {"proj": layer})

    loaded = load_adapter(path, {"proj": base})
    loaded_layer = loaded["proj"]

    x = torch.randn(3, 16)
    with torch.no_grad():
        original_out = layer(x)
        loaded_out = loaded_layer(x)
    assert torch.allclose(original_out, loaded_out, atol=1e-6)


def test_multi_layer_adapter_round_trip(tmp_path):
    bases = {}
    layers = {}
    for name in ("q_proj", "v_proj"):
        base, layer = _trained_layer(rank=4)
        bases[name] = base
        layers[name] = layer
    path = tmp_path / "adapter.safetensors"

    save_adapter(path, layers)
    loaded = load_adapter(path, bases)

    assert set(loaded.keys()) == {"q_proj", "v_proj"}
    for name in loaded:
        assert torch.equal(loaded[name].plugin_matrix_A, layers[name].plugin_matrix_A)


def test_save_adapter_rejects_empty_layers(tmp_path):
    with pytest.raises(ValueError):
        save_adapter(tmp_path / "adapter.safetensors", {})


def test_load_adapter_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_adapter(tmp_path / "does-not-exist.safetensors", {"proj": nn.Linear(4, 4)})


def test_load_adapter_mismatched_layer_names_raises(tmp_path):
    base, layer = _trained_layer()
    path = tmp_path / "adapter.safetensors"
    save_adapter(path, {"proj": layer})

    with pytest.raises(KeyError):
        load_adapter(path, {"wrong_name": nn.Linear(16, 8)})


def test_save_creates_parent_directories(tmp_path):
    base, layer = _trained_layer()
    nested_path = tmp_path / "nested" / "dirs" / "adapter.safetensors"
    save_adapter(nested_path, {"proj": layer})
    assert nested_path.exists()

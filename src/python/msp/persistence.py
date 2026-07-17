"""
Adapter persistence: save/load a set of StructuralPluginLayer weights to a
single .safetensors file, so adapters can actually be distributed and
reloaded rather than only existing for the lifetime of a Python process.

This addresses STATUS.md's "no save/load format for adapter weights"
gap. safetensors was chosen (over e.g. raw torch.save/pickle) because:

  - It's the format the LoRA/PEFT ecosystem this project is otherwise
    aligned with already uses, so adapters produced here are immediately
    usable by/compatible with that tooling's expectations.
  - It does not execute arbitrary code on load (unlike pickle-based
    torch.save), which matters for a format meant to carry
    third-party-published adapters (the B2B marketplace use case) -- a
    malicious .safetensors file cannot achieve code execution just by
    being loaded, only a malicious *pickle* file can.
  - It stores metadata (rank, alpha, dtype) alongside the tensors in a
    single file, so loading an adapter doesn't require a second sidecar
    file to reconstruct it correctly.

This module intentionally does NOT verify integrity or authenticity of a
loaded file -- pair it with `src/daemon/integrity_check.c`'s
msp_verify_integrity/msp_verify_signature (via a future Python binding,
see STATUS.md) if loading adapters from an untrusted source. Silently
trusting file contents is fine for the local save/load round-trip this
module is tested against; it is NOT fine for adapters obtained from a
marketplace over the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from .plugin_layer import StructuralPluginLayer

_METADATA_KEY = "__msp_adapter_metadata__"


def save_adapter(path: Union[str, Path], layers: Dict[str, StructuralPluginLayer]) -> None:
    """
    Save every layer's plugin_matrix_A/B, plus enough metadata (rank,
    alpha, routing_gate_enabled) to reconstruct each StructuralPluginLayer
    exactly, into a single .safetensors file at `path`.

    Does NOT save the frozen base_layer weights -- those belong to the
    foundation model this adapter is meant to sit on top of, not to the
    adapter itself, matching the whole point of a parameter-efficient
    adapter (it should be tiny relative to the base model).
    """
    if not layers:
        raise ValueError("save_adapter: layers must be non-empty")

    tensors: Dict[str, torch.Tensor] = {}
    per_layer_metadata = {}

    for layer_name, layer in layers.items():
        if not isinstance(layer, StructuralPluginLayer):
            raise TypeError(
                f"layer '{layer_name}' is a {type(layer).__name__}, "
                f"not a StructuralPluginLayer"
            )
        # safetensors requires contiguous tensors and does not accept
        # tensors that share underlying storage with another saved
        # tensor -- .contiguous().clone() guarantees both, at the cost of
        # a small, one-time copy at save time.
        tensors[f"{layer_name}.plugin_matrix_A"] = layer.plugin_matrix_A.detach().contiguous().clone()
        tensors[f"{layer_name}.plugin_matrix_B"] = layer.plugin_matrix_B.detach().contiguous().clone()
        per_layer_metadata[layer_name] = {
            "rank": layer.rank,
            # alpha isn't stored directly on the layer (only the derived
            # `scaling = alpha / rank` is) -- recover it exactly rather
            # than storing a float that could round-trip imprecisely.
            "alpha": layer.scaling * layer.rank,
            "routing_gate_enabled": layer.routing_gate_enabled,
        }

    # safetensors' own metadata dict must be Dict[str, str] -- serialize
    # our structured metadata as a single JSON-encoded string value rather
    # than fighting that constraint with a flatter, lossier encoding.
    metadata = {_METADATA_KEY: json.dumps(per_layer_metadata)}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def load_adapter(
    path: Union[str, Path],
    base_layers: Dict[str, nn.Linear],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, StructuralPluginLayer]:
    """
    Reconstruct StructuralPluginLayer instances from a file written by
    save_adapter(), wrapping the given `base_layers` (the frozen
    foundation-model layers this adapter attaches to -- these are NOT
    part of the saved file, see save_adapter's docstring).

    Raises KeyError if a layer name present in the saved file has no
    corresponding entry in `base_layers`, or vice versa for a layer this
    adapter doesn't actually cover -- a silent partial load would be
    worse than an explicit error here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"adapter file not found: {path}")

    tensors = load_file(str(path), device=str(device) if device is not None else "cpu")

    # safetensors' load_file doesn't return the header's metadata dict
    # directly -- re-open just the header to get it. This is cheap (a
    # small JSON header read, not the tensor data) compared to loading
    # tensors twice.
    metadata = _read_metadata(path)
    if metadata is None or _METADATA_KEY not in metadata:
        raise ValueError(
            f"'{path}' has no MSP adapter metadata -- was it written by "
            f"msp.persistence.save_adapter?"
        )
    per_layer_metadata = json.loads(metadata[_METADATA_KEY])

    saved_layer_names = set(per_layer_metadata.keys())
    requested_layer_names = set(base_layers.keys())
    if saved_layer_names != requested_layer_names:
        missing = saved_layer_names - requested_layer_names
        extra = requested_layer_names - saved_layer_names
        raise KeyError(
            f"base_layers do not match the saved adapter's layer names. "
            f"Missing base_layers for: {sorted(missing) or 'none'}; "
            f"base_layers with no saved data: {sorted(extra) or 'none'}"
        )

    result: Dict[str, StructuralPluginLayer] = {}
    for layer_name, base_layer in base_layers.items():
        meta = per_layer_metadata[layer_name]
        layer = StructuralPluginLayer(
            base_layer,
            rank=meta["rank"],
            alpha=meta["alpha"],
        )
        with torch.no_grad():
            layer.plugin_matrix_A.copy_(tensors[f"{layer_name}.plugin_matrix_A"])
            layer.plugin_matrix_B.copy_(tensors[f"{layer_name}.plugin_matrix_B"])
        layer.routing_gate_enabled = meta["routing_gate_enabled"]
        result[layer_name] = layer

    return result


def _read_metadata(path: Path) -> Optional[Dict[str, str]]:
    """
    Reads just the safetensors header (a small JSON blob at the start of
    the file, prefixed by its own byte length) to recover the metadata
    dict, without loading any tensor data.
    """
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return None
        header_len = int.from_bytes(header_len_bytes, byteorder="little")
        header_json = f.read(header_len)
        header = json.loads(header_json)
    return header.get("__metadata__")

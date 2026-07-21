"""
End-to-end example: wiring StructuralPluginLayer into a real multi-layer
transformer, training two independent low-rank adapters on top of one
frozen base, and exercising AdapterManager (memory budget + hot-swap),
ThermalMonitor (gradient gating), and msp.persistence (save/reload) the
way a real caller would use them together -- not in isolation.

This is docs/STATUS.md's "what's left to do" item #2: everything in
`msp` is unit-tested one piece at a time, but nothing before this file
wired a StructuralPluginLayer into an actual multi-layer transformer
block and ran a real training loop against it.

Deliberately scoped:
- The synthetic task (a per-token modular-arithmetic shift) is chosen so
  the example trains fast and deterministically on CPU, not to be a
  realistic language-modeling task. What matters here is the wiring, not
  the task's difficulty.
- Only `out_proj`, `fc1`, and `fc2` of each transformer block are wrapped
  in StructuralPluginLayer adapters (a common minimal PEFT target-module
  choice); `q_proj`/`k_proj` are left frozen along with everything else
  in the base. Adapting them too is a one-line change to
  ADAPTED_SUBLAYERS below.
- This does NOT wire in the CUDA kernel (src/cuda/) or
  sandbox_watchdog's fallback/control signal (src/daemon/) -- see
  docs/STATUS.md's "what's left to do" items #1 and #4 for why those
  are separate, larger pieces of work. This file only gives them
  something real to eventually plug into on the Python training-loop
  side.

Run directly:
    PYTHONPATH=src/python python examples/e2e_training.py

Or import the individual pieces -- as
tests/python/test_e2e_training_example.py does -- to build smaller,
faster, more targeted scenarios.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from msp import (
    AdapterBudgetError,
    AdapterManager,
    StructuralPluginLayer,
    ThermalMonitor,
    load_adapter,
    save_adapter,
)
from msp.thermal import ScriptedThermalReader

# Which sublayers of each TinyTransformerBlock get wrapped in an adapter.
# q_proj/k_proj are intentionally left out -- see module docstring.
ADAPTED_SUBLAYERS = ("out_proj", "fc1", "fc2")


class TinyTransformerBlock(nn.Module):
    """
    A standard pre-LN transformer encoder block (multi-head self-attention
    + feed-forward, each with a residual connection), sized small enough
    to train in well under a second per step on CPU.

    q_proj/k_proj/v_proj/out_proj are kept as separate nn.Linear modules
    (rather than one fused in_proj, as e.g. nn.MultiheadAttention uses)
    specifically so individual sublayers can be wrapped in a
    StructuralPluginLayer independently -- fusing them would make
    per-projection adaptation impossible without unpacking the fused
    weight first.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.activation = nn.GELU()

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attn_out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(attn_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._self_attention(self.ln1(x))
        hidden = self.ln2(x)
        hidden = self.fc2(self.activation(self.fc1(hidden)))
        return x + hidden


def freeze_all_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def build_frozen_base(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    ffn_dim: int,
    n_layers: int,
) -> tuple[nn.Embedding, nn.ModuleList, nn.Linear]:
    """
    Construct the "frozen foundation model" this example's adapters sit
    on top of: an embedding, a stack of TinyTransformerBlocks, and an
    output head projecting back to vocab logits. Every parameter is
    frozen here, before any adapter is attached -- adaptation happens
    exclusively through the low-rank matrices StructuralPluginLayer adds.
    """
    embed = nn.Embedding(vocab_size, d_model)
    blocks = nn.ModuleList([TinyTransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)])
    head = nn.Linear(d_model, vocab_size)

    freeze_all_parameters(embed)
    freeze_all_parameters(blocks)
    freeze_all_parameters(head)

    return embed, blocks, head


def collect_base_sublayers(blocks: nn.ModuleList) -> Dict[str, nn.Linear]:
    """
    Snapshot of the frozen base nn.Linear objects each adapter attaches
    to, keyed "block{i}.{sublayer_name}" -- the same keying
    AdapterManager and msp.persistence use.

    Call this once, right after build_frozen_base and before any adapter
    is plugged in via plug_in(); keep the result around, since
    persistence.load_adapter needs this exact dict later to reconstruct
    an adapter for these same base layers.
    """
    base_linears: Dict[str, nn.Linear] = {}
    for i, blk in enumerate(blocks):
        for sublayer_name in ADAPTED_SUBLAYERS:
            linear = getattr(blk, sublayer_name)
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"blocks[{i}].{sublayer_name} is a {type(linear).__name__}, not "
                    "nn.Linear -- collect_base_sublayers must be called before any "
                    "adapter has been plugged in via plug_in()"
                )
            base_linears[f"block{i}.{sublayer_name}"] = linear
    return base_linears


def attach_adapters(
    base_linears: Dict[str, nn.Linear], rank: int, alpha: float
) -> Dict[str, StructuralPluginLayer]:
    """
    Build a fresh, independent set of StructuralPluginLayer adapters
    wrapping `base_linears` (see collect_base_sublayers). Calling this
    twice against the same base_linears produces two independent
    adapters over the same frozen base -- the "one base, many swappable
    adapters" story this project is about.
    """
    return {name: StructuralPluginLayer(linear, rank=rank, alpha=alpha) for name, linear in base_linears.items()}


def plug_in(blocks: nn.ModuleList, layers: Dict[str, StructuralPluginLayer]) -> None:
    """Make `layers` the live adapter driving each block's forward pass."""
    for i, blk in enumerate(blocks):
        for sublayer_name in ADAPTED_SUBLAYERS:
            key = f"block{i}.{sublayer_name}"
            if key in layers:
                setattr(blk, sublayer_name, layers[key])


def forward(embed: nn.Embedding, blocks: nn.ModuleList, head: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
    x = embed(tokens)
    for blk in blocks:
        x = blk(x)
    return head(x)


def synthetic_batch(
    vocab_size: int,
    batch: int,
    seq_len: int,
    shift: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A deterministic, trivially-learnable synthetic task: predict each
    input token shifted by a fixed amount, mod vocab_size. Different
    `shift` values stand in for different "tasks" a different adapter has
    learned -- see main() below for training two adapters on two shifts
    against the same frozen base.
    """
    tokens = torch.randint(0, vocab_size, (batch, seq_len), generator=generator)
    targets = (tokens + shift) % vocab_size
    return tokens, targets


def train_adapter(
    embed: nn.Embedding,
    blocks: nn.ModuleList,
    head: nn.Linear,
    adapter_layers: Dict[str, StructuralPluginLayer],
    *,
    vocab_size: int,
    shift: int,
    steps: int = 150,
    batch: int = 16,
    seq_len: int = 4,
    lr: float = 1e-2,
    thermal_monitor: Optional[ThermalMonitor] = None,
    generator: Optional[torch.Generator] = None,
) -> List[float]:
    """
    A real training loop: samples a fresh synthetic batch every step and
    does a full forward/backward/optimizer.step() through the whole
    `blocks` stack (not a single layer in isolation), returning the
    per-step loss history.

    Only `adapter_layers`' own plugin_matrix_A/B parameters are ever
    passed to the optimizer -- the frozen base (embed, blocks' own
    q_proj/k_proj, head) never receives a gradient update, matching this
    project's whole premise.

    `plug_in(blocks, adapter_layers)` must be called before this (it is
    not done here) so the caller stays explicitly in control of which
    adapter is currently live in the forward path.

    If `thermal_monitor` is given, gradient gating is armed on every
    layer in `adapter_layers` for the duration of this call (and cleared
    again before returning, so a reused monitor/layer combination never
    leaves stale hooks registered).
    """
    if thermal_monitor is not None:
        for layer in adapter_layers.values():
            layer.gate_gradients(thermal_monitor)

    params = [p for layer in adapter_layers.values() for p in (layer.plugin_matrix_A, layer.plugin_matrix_B)]
    optimizer = torch.optim.Adam(params, lr=lr)

    losses: List[float] = []
    try:
        for _ in range(steps):
            tokens, targets = synthetic_batch(vocab_size, batch, seq_len, shift, generator=generator)
            logits = forward(embed, blocks, head, tokens)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    finally:
        if thermal_monitor is not None:
            for layer in adapter_layers.values():
                layer.clear_gradient_gate()

    return losses


def main() -> None:
    torch.manual_seed(0)
    vocab_size, d_model, n_heads, ffn_dim, n_layers = 12, 32, 4, 64, 2
    rank, alpha = 8, 16.0

    embed, blocks, head = build_frozen_base(vocab_size, d_model, n_heads, ffn_dim, n_layers)
    base_linears = collect_base_sublayers(blocks)

    layers_a = attach_adapters(base_linears, rank, alpha)
    layers_b = attach_adapters(base_linears, rank, alpha)

    one_adapter_bytes = sum(layer.parameter_bytes() for layer in layers_a.values())
    print(f"each adapter: {one_adapter_bytes} bytes across {len(layers_a)} layers (rank={rank})")

    # Budget sized to fit exactly one resident adapter at a time, so
    # loading the second while the first is still resident is rejected --
    # a real (not synthetic) exercise of AdapterManager's budget check,
    # sized from these adapters' actual parameter_bytes().
    manager = AdapterManager(memory_budget_bytes=int(one_adapter_bytes * 1.5))
    manager.load_adapter("task-a-shift1", layers_a)
    manager.activate("task-a-shift1")

    print("\n--- training task-a (shift = 1) ---")
    plug_in(blocks, layers_a)
    losses_a = train_adapter(embed, blocks, head, layers_a, vocab_size=vocab_size, shift=1)
    print(f"loss: {losses_a[0]:.4f} -> {losses_a[-1]:.6f} over {len(losses_a)} steps")

    print("\nattempting to load task-b while task-a is still resident (expected: rejected)")
    try:
        manager.load_adapter("task-b-shift3", layers_b)
        raise AssertionError("expected AdapterBudgetError, but load_adapter succeeded")
    except AdapterBudgetError as exc:
        print(f"  -> AdapterBudgetError, as expected: {exc}")

    print("\n--- saving task-a to disk, then evicting it to make room ---")
    save_dir = Path(tempfile.mkdtemp(prefix="msp_e2e_"))
    save_path = save_dir / "task-a-shift1.safetensors"
    save_adapter(save_path, layers_a)
    manager.unload_adapter("task-a-shift1")
    plug_in(blocks, layers_b)  # nothing still references layers_a's tensors after this
    del layers_a
    print(f"saved to {save_path} ({save_path.stat().st_size} bytes)")
    print(f"manager resident bytes after eviction: {manager.resident_bytes}")

    print("\n--- training task-b (shift = 3), now that there's budget ---")
    manager.load_adapter("task-b-shift3", layers_b)
    manager.activate("task-b-shift3")
    losses_b = train_adapter(embed, blocks, head, layers_b, vocab_size=vocab_size, shift=3)
    print(f"loss: {losses_b[0]:.4f} -> {losses_b[-1]:.6f} over {len(losses_b)} steps")

    print("\n--- fine-tuning task-b under a simulated hot spell (gradient gating) ---")
    hot_then_cool = ScriptedThermalReader([80.0] * 20 + [25.0] * 20)
    monitor = ThermalMonitor(reader=hot_then_cool, freeze_threshold_c=75.0, freeze_ratio=0.2)
    gated_losses = train_adapter(
        embed, blocks, head, layers_b,
        vocab_size=vocab_size, shift=3, steps=40, thermal_monitor=monitor,
    )
    print(f"loss under thermal gating: {gated_losses[0]:.6f} -> {gated_losses[-1]:.6f}")

    print("\n--- reloading task-a from disk and confirming it serves again ---")
    layers_a_reloaded = load_adapter(save_path, base_layers=base_linears)
    manager.unload_adapter("task-b-shift3")  # evict to make room, same as before
    plug_in(blocks, layers_a_reloaded)
    manager.load_adapter("task-a-shift1-reloaded", layers_a_reloaded)
    manager.activate("task-a-shift1-reloaded")

    tokens, _ = synthetic_batch(vocab_size, batch=8, seq_len=4, shift=1)
    with torch.no_grad():
        reloaded_logits = forward(embed, blocks, head, tokens)
    print(
        f"reloaded task-a logits shape: {tuple(reloaded_logits.shape)}, "
        f"finite: {bool(torch.isfinite(reloaded_logits).all())}"
    )

    shutil.rmtree(save_dir)
    print("\ndone.")


if __name__ == "__main__":
    main()

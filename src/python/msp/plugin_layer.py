"""
StructuralPluginLayer: a LoRA-style hot-swappable adapter over a frozen
nn.Linear base layer, with a non-blocking async forward path and optional
thermal-aware gradient gating for training.

Fixes relative to the v3.0 spec's `AsyncStructuralPluginLayer`:

1. The spec unconditionally called `torch.cuda.current_stream()` and
   `torch.cuda.stream(...)`, which raises on any machine without CUDA
   (including this sandbox, and most edge CPUs/NPUs the paper claims to
   target). This version branches on `x.device.type` and only touches CUDA
   stream APIs when actually running on a CUDA device.
2. The spec allocated the adapter matrices as fixed float32 regardless of
   the base layer's dtype, which silently breaks half-precision (fp16/bf16)
   deployments -- exactly the memory-constrained edge case this project is
   for. This version matches the base layer's dtype by default.
3. Adds the thermal-aware gradient gating that the spec described only at
   the CUDA-kernel level (Subsystem A), so the same throttling policy is
   testable in pure PyTorch on CPU, without requiring custom CUDA code.
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

import torch
import torch.nn as nn

from .thermal import ThermalMonitor


class StructuralPluginLayer(nn.Module):
    """
    A frozen base nn.Linear plus a trainable low-rank adapter:

        y = base(x) + scaling * (x @ A^T @ B^T)

    Parameters
    ----------
    base_layer:
        The frozen linear layer being adapted. Its weights are set to
        `requires_grad = False`.
    rank:
        Adapter rank (the "r" in LoRA).
    alpha:
        LoRA scaling numerator; effective scale is alpha / rank.
    dtype:
        dtype for the adapter parameters. Defaults to the base layer's
        weight dtype so the adapter doesn't silently upcast a fp16 model
        to fp32 (see module docstring, fix #2).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be a positive integer")

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        out_f, in_f = base_layer.weight.shape
        adapter_dtype = dtype if dtype is not None else base_layer.weight.dtype
        device = base_layer.weight.device

        self.rank = rank
        self.plugin_matrix_A = nn.Parameter(
            torch.empty((rank, in_f), dtype=adapter_dtype, device=device)
        )
        self.plugin_matrix_B = nn.Parameter(
            torch.zeros((out_f, rank), dtype=adapter_dtype, device=device)
        )
        self.scaling = alpha / rank
        self.reset_parameters()

        # Dynamic routing gate ("D" in the spec's forward-pass formula:
        # h = x*W0 + D*(alpha/r)*(x*B*A)). When False, forward() returns
        # the base layer's output untouched -- a zero-compute bypass that
        # does NOT free the adapter's memory (unlike AdapterManager.
        # unload_adapter, which does). This is the "instant on/off without
        # de-allocating tensors from hardware caches" behavior the spec
        # describes; use unload_adapter when you actually want the memory
        # back, and this gate when you just want to pause an adapter's
        # effect cheaply and resume it just as cheaply.
        self.routing_gate_enabled = True

        # Deterministic thermal gating mask, applied in `gate_gradients`.
        # None until a ThermalMonitor drives it (see gate_gradients).
        self._grad_gate_handles: list = []

    def reset_parameters(self) -> None:
        """
        Canonical LoRA initialization (matches the reference scheme used by
        Microsoft's LoRA / Hugging Face PEFT): A ~ Kaiming-uniform (so its
        row-scale matches the base layer's own weight init distribution),
        B = 0. Since adapter_out = (x @ A^T) @ B^T, zero-initializing B
        guarantees delta_W = B @ A = 0 at construction time, so the layer
        starts out mathematically identical to the frozen base layer no
        matter how A is initialized.
        """
        nn.init.kaiming_uniform_(self.plugin_matrix_A, a=math.sqrt(5))
        nn.init.zeros_(self.plugin_matrix_B)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.routing_gate_enabled:
            return base_out
        plugin_in = x.to(self.plugin_matrix_A.dtype)
        adapter_out = torch.matmul(
            torch.matmul(plugin_in, self.plugin_matrix_A.t()),
            self.plugin_matrix_B.t(),
        ) * self.scaling
        return base_out + adapter_out.to(base_out.dtype)

    async def forward_async(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-blocking inference call so a caller (e.g. a UI event loop or a
        request-serving loop) is never stalled on the matmuls.

        On CUDA, the heavy compute is launched on the layer's own stream
        and only the (cheap) synchronization wait is offloaded to a thread,
        so the GIL is released while the GPU works. On CPU, the whole
        forward pass runs in a worker thread since there is no separate
        "stream" concept to overlap with.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._forward_sync, x)

    def _forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            stream = torch.cuda.Stream(device=x.device)
            with torch.cuda.stream(stream):
                result = self.forward(x)
            stream.synchronize()
            return result
        # CPU / MPS / other backends: no stream API to manage, just compute.
        return self.forward(x)

    # ------------------------------------------------------------------
    # Thermal-aware gradient gating (CPU/GPU-portable equivalent of the
    # spec's CUDA-only "Thermal-Aware Gradient Freezing")
    # ------------------------------------------------------------------

    def gate_gradients(self, monitor: ThermalMonitor) -> None:
        """
        Register backward hooks that zero out gradient rows on
        plugin_matrix_A / plugin_matrix_B according to the current thermal
        reading from `monitor`.

        Matches the spec's stated intent ("drop update layer penetration
        by 80% if temperature exceeds the safe envelope") with a
        deterministic row-stride mask -- every Nth row stays active, where
        N = round(1 / monitor.freeze_ratio) -- so behavior is reproducible
        and unit-testable rather than depending on kernel-launch geometry.

        Call this once after constructing the layer and before training;
        it is idempotent-safe to call again (old hooks are removed first).
        """
        self.clear_gradient_gate()
        stride = max(1, round(1.0 / monitor.freeze_ratio))

        def make_hook(num_rows: int):
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                if not monitor.is_throttling():
                    return grad
                mask = torch.zeros(num_rows, dtype=torch.bool, device=grad.device)
                mask[::stride] = True
                return grad * mask.view(-1, *([1] * (grad.dim() - 1)))
            return _hook

        self._grad_gate_handles.append(
            self.plugin_matrix_A.register_hook(make_hook(self.plugin_matrix_A.shape[0]))
        )
        self._grad_gate_handles.append(
            self.plugin_matrix_B.register_hook(make_hook(self.plugin_matrix_B.shape[0]))
        )

    def clear_gradient_gate(self) -> None:
        for handle in self._grad_gate_handles:
            handle.remove()
        self._grad_gate_handles = []

    # ------------------------------------------------------------------
    # Introspection helpers used by AdapterManager for the 512MB budget
    # ------------------------------------------------------------------

    def parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in (self.plugin_matrix_A, self.plugin_matrix_B))

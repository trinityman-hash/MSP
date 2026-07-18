"""
Numerically validate src/cuda/fused_msp_backward_kernel.cu on a real
NVIDIA GPU.

This is the "concrete next step" flagged in docs/STATUS.md's "CUDA
validation on Google Colab" section: compiling the kernel (already done
separately with nvcc) only proves it *builds* -- it says nothing about
whether the math or the thermal-gating logic is correct. This script is
the missing piece: it runs the actual kernel on the GPU and diffs its
output against a trusted PyTorch autograd reference computed from
msp.plugin_layer.StructuralPluginLayer's own forward pass.

What it checks
---------------
1. Correctness (no throttling): compiles
   `msp_gradient_A_thermal_gated_kernel` via NVRTC (through cupy), runs it
   on a small fixed input (batch=4, in_features=16, rank=4 -- the sizes
   STATUS.md itself suggests), and compares every element of grad_A
   against torch.autograd's gradient for the same input, obtained by
   constructing an actual StructuralPluginLayer and backward()-ing through
   u = x @ A^T with a fixed upstream gradient (see `_reference_grad_A`
   for the derivation).
2. Correctness (throttling engaged): re-runs with current_temp_c above
   critical_temp_c and freeze_stride=2, and checks two things at once --
   that the rows which *do* fire still match the reference exactly, and
   that the rows which should be skipped are left completely untouched in
   device memory (per the kernel's own documented contract: "no global
   memory touched for this (row, col) at all"). grad_A is pre-filled with
   a sentinel value before launch specifically so an accidental write to a
   frozen row is caught rather than silently matching.

Prerequisites (run these as their own Colab cell first if not already
present -- recent Colab images often already have cupy preinstalled, so
try running this script before installing anything):

    !pip install -q cupy-cuda12x   # or cupy-cuda13x -- match your CUDA
                                    # toolkit; check with `!nvcc --version`
                                    # if the first one fails to import.

Usage (from the repo root, e.g. right after `%cd MSP` in Colab):

    !python tests/cuda/validate_gradient_kernel.py

Exits 0 if every check passes, 1 otherwise, so it's safe to depend on the
exit code from a notebook cell or CI step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = REPO_ROOT / "src" / "cuda" / "fused_msp_backward_kernel.cu"
KERNEL_NAME = "msp_gradient_A_thermal_gated_kernel"

sys.path.insert(0, str(REPO_ROOT / "src" / "python"))


def _extract_kernel_source(full_source: str) -> str:
    """
    Pulls out just the `extern "C" { ... }` block containing the
    __global__ kernel, via brace counting. Deliberately leaves out the
    host-side wrapper (msp_launch_gradient_A_kernel and its
    cudaError_t/fprintf/cstdio usage) below it, since that's host-only
    code NVRTC (the runtime compiler cupy uses) isn't set up to compile --
    NVRTC wants just the device kernel itself.

    Reads the kernel straight from the real file on every run (rather than
    embedding a copy of the source in this script) so this test always
    exercises whatever is currently checked in, not a stale snapshot.
    """
    marker = 'extern "C" {'
    start = full_source.index(marker)
    body_start = start + len(marker)
    depth = 1
    i = body_start
    while depth > 0:
        c = full_source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    kernel_body = full_source[body_start : i - 1]
    return f'extern "C" {{\n{kernel_body}\n}}\n'


def _reference_grad_A(x: torch.Tensor, grad_u: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Trusted reference value, via actual autograd through the same
    u = x @ A^T relationship StructuralPluginLayer uses internally.

    loss = sum(u * grad_u) is the standard trick for injecting an
    arbitrary, chosen upstream gradient: d(loss)/du == grad_u exactly, so
    A.grad after backward() is the real dL/dA for that upstream gradient
    -- precisely what the kernel computes, and exactly what a prior
    kernel/op in a real training step would hand it.
    """
    A = A.detach().clone().requires_grad_(True)
    u = x @ A.t()
    loss = (u * grad_u).sum()
    loss.backward()
    return A.grad.detach().clone()


def _run_kernel(cp, kernel, x_np, grad_u_np, batch, in_f, rank,
                 current_temp_c, critical_temp_c, freeze_stride, sentinel):
    x_dev = cp.asarray(x_np)
    grad_u_dev = cp.asarray(grad_u_np)
    grad_A_dev = cp.full((rank, in_f), sentinel, dtype=cp.float32)

    threads_per_block = 256
    block = (threads_per_block, 1, 1)
    grid = ((in_f + threads_per_block - 1) // threads_per_block, rank, 1)

    kernel(
        grid, block,
        (
            x_dev, grad_u_dev, grad_A_dev,
            np.int32(batch), np.int32(in_f), np.int32(rank),
            np.float32(current_temp_c), np.float32(critical_temp_c),
            np.int32(freeze_stride),
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(grad_A_dev)


def main() -> int:
    print(f"Reading kernel from {KERNEL_PATH.relative_to(REPO_ROOT)}")
    if not KERNEL_PATH.exists():
        print(f"FAIL: kernel file not found at {KERNEL_PATH}")
        return 1
    full_source = KERNEL_PATH.read_text()
    kernel_source = _extract_kernel_source(full_source)

    try:
        import cupy as cp
    except ImportError:
        print(
            "FAIL: cupy is not installed. Run this in its own cell first:\n"
            "  !pip install -q cupy-cuda12x\n"
            "(if that wheel fails to import, check `!nvcc --version` and try "
            "cupy-cuda13x instead to match your CUDA toolkit version)"
        )
        return 1

    if cp.cuda.runtime.getDeviceCount() == 0:
        print(
            "FAIL: no CUDA device visible to cupy. In Colab: Runtime -> "
            "Change runtime type -> T4 GPU -> Save, then re-run."
        )
        return 1
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU: {device_name}")

    try:
        module = cp.RawModule(code=kernel_source)
        kernel = module.get_function(KERNEL_NAME)
    except Exception as exc:  # noqa: BLE001 -- want to show the NVRTC error verbatim
        print(f"FAIL: kernel failed to compile via NVRTC:\n{exc}")
        print("\n--- extracted kernel source (for debugging) ---")
        print(kernel_source)
        return 1
    print("Kernel compiled via NVRTC OK.")

    from msp.plugin_layer import StructuralPluginLayer

    torch.manual_seed(0)
    batch, in_f, rank = 4, 16, 4
    base = nn.Linear(in_f, in_f // 2)
    layer = StructuralPluginLayer(base, rank=rank, alpha=8.0)

    x = torch.randn(batch, in_f)
    grad_u = torch.randn(batch, rank)  # stand-in upstream gradient dL/du
    A = layer.plugin_matrix_A

    x_np = x.numpy().astype(np.float32)
    grad_u_np = grad_u.numpy().astype(np.float32)

    reference = _reference_grad_A(x, grad_u, A).numpy().astype(np.float32)

    sentinel = -12345.0
    all_ok = True

    # --- Check 1: no throttling, every row should fire and match. ---
    cool_out = _run_kernel(
        cp, kernel, x_np, grad_u_np, batch, in_f, rank,
        current_temp_c=50.0, critical_temp_c=75.0, freeze_stride=2,
        sentinel=sentinel,
    )
    cool_ok = np.allclose(cool_out, reference, atol=1e-4, rtol=1e-4)
    print(f"[no throttling]  matches PyTorch autograd reference: {cool_ok}")
    if not cool_ok:
        print("  max abs diff:", np.abs(cool_out - reference).max())
    all_ok &= cool_ok

    # --- Check 2: throttling engaged, freeze_stride=2 -> rows 0,2 active. ---
    hot_out = _run_kernel(
        cp, kernel, x_np, grad_u_np, batch, in_f, rank,
        current_temp_c=90.0, critical_temp_c=75.0, freeze_stride=2,
        sentinel=sentinel,
    )
    active_rows = [r for r in range(rank) if r % 2 == 0]
    frozen_rows = [r for r in range(rank) if r % 2 != 0]

    active_ok = np.allclose(hot_out[active_rows], reference[active_rows], atol=1e-4, rtol=1e-4)
    frozen_ok = np.all(hot_out[frozen_rows] == sentinel)
    print(f"[throttling]     active rows {active_rows} match reference: {active_ok}")
    print(f"[throttling]     frozen rows {frozen_rows} left untouched (sentinel intact): {frozen_ok}")
    if not active_ok:
        print("  max abs diff on active rows:", np.abs(hot_out[active_rows] - reference[active_rows]).max())
    if not frozen_ok:
        print("  frozen-row values (expected all == sentinel):", hot_out[frozen_rows])
    all_ok &= active_ok & frozen_ok

    print()
    if all_ok:
        print("PASS: fused_msp_backward_kernel.cu matches the PyTorch reference "
              "(both throttled and unthrottled), on real GPU hardware.")
        return 0
    else:
        print("FAIL: see diffs above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

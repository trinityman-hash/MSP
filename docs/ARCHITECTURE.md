# MSP Architecture

This document explains how this repository is structured, and — since the
original design docs (`Project Apex-Edge`, v2 and v3.0) contained several
real technical defects — what was changed relative to those docs and why.
If you only read one section, read "What was fixed."

## What this project is

Modular Structural Plugins (MSPs): hot-swappable, LoRA-style low-rank
adapters that sit alongside a frozen base model's projection layers, with:

- a dynamic on/off gate (`D` in the spec's math) for zero-compute bypass
  without freeing memory,
- an adapter registry that enforces a memory budget across all resident
  adapters and evicts an adapter's entire footprint (all its layers, all
  its KV-cache blocks) in one call,
- thermal-aware gradient throttling during on-device training, and
- a watchdog that can force a fast rollback to base-model-only behavior if
  telemetry (temperature, memory, a security check) crosses a threshold.

## Layout

```
src/python/msp/       Reference implementation (PyTorch). Portable,
                       CPU-testable, the source of truth for behavior.
src/cpp/               EloraAllocator: adapter-scoped KV-cache allocator.
src/cuda/               NVIDIA CUDA kernel for on-device gradient compute.
src/daemon/             C watchdog + integrity-check daemon.
tests/                  Mirrors src/ layout. Python via pytest, C/C++ via
                        plain executables run through ctest (no external
                        test framework dependency).
docs/                   This file, plus STATUS.md and SECURITY.md.
```

## What was fixed relative to the original spec

The v3.0 spec (the detailed systems doc) and v2 spec (the "Apex-Edge"
architecture/market doc) disagreed with each other in places and each
contained defects on their own. In rough order of severity:

### 1. Cross-thread `longjmp` (undefined behavior)

The v3.0 watchdog saved a `jmp_buf` on the monitored (inference) thread and
called `longjmp()` to it from the watchdog thread. POSIX only guarantees
`setjmp`/`longjmp` for returning within the *same* thread's call stack;
jumping across threads corrupts the target thread's stack. It tends to
"work" in casual testing and fail under load — exactly when a
thermal/security fallback needs to actually fire.

**Fix:** the monitored thread arms a `sigsetjmp` point itself; the
watchdog signals it via `pthread_kill()`, and `siglongjmp()` runs inside
the signal handler, in the interrupted thread's own context. See
`src/daemon/sandbox_watchdog.h` for the full writeup, including a second,
more subtle version of the same bug that showed up in the first draft of
this fix (wrapping `sigsetjmp` in a plain function that returns before the
jump can safely target it) — worth reading if you're touching this file,
since it's a real trap.

### 2. Single CUDA kernel targeting three incompatible platforms

The v3.0 spec asked to compile one `.cu` file for "Apple Metal" and
"Google Tensor TPU VM IR" as well as CUDA. CUDA is NVIDIA-only; it cannot
target Metal or a mobile NPU. `src/cuda/fused_msp_backward_kernel.cu` is
explicitly the CUDA/NVIDIA backend only.

The v2 spec's pipeline — raw weights → ONNX graph optimization → Apache
TVM/LLVM → target-specific codegen (Apple AMX / Qualcomm Hexagon NPU) — is
the architecturally sound way to solve the actual cross-platform problem,
and is the one worth building toward. It is not implemented here: it needs
the TVM toolchain and vendor SDKs (Xcode toolchain, Qualcomm Hexagon SDK),
neither of which is available in the environment this repo was built in.
Treat `src/cuda/` as one backend among several that a real multi-platform
build would need, not as portable across hardware on its own.

### 3. "RSA-4096 hash validation" (category error)

RSA is a signature/encryption primitive, not a hash function. Integrity
(has this data been tampered with?) and authenticity (did this really come
from a trusted publisher?) are different properties, checked with
different primitives. `src/daemon/integrity_check.c` implements the
integrity half concretely with SHA-256, and documents the authenticity
half (a signature over that hash) as a deliberate stub — wiring it up for
real requires a trust-root decision (embedded public key vs. certificate
chain vs. attestation service) that's a deployment choice, not a library
default.

### 4. "Ring 3 Security Sandbox" overstates what `mmap` provides

Ring 3 is CPU privilege-level terminology for unprivileged execution —
ordinary userspace code already runs there; a userspace `mmap()` call
doesn't grant or control it. `msp_allocate_isolated_region()` does provide
a real property (no `PROT_EXEC`, so data mapped there can't be directly
executed), but that alone is not process isolation. See `SECURITY.md` for
what real isolation of untrusted adapter code would need.

### 5. CUDA-only allocator (didn't compile without the CUDA toolkit)

The v3.0 `EloraAllocator` called `cudaMallocManaged`/`cudaFree`
unconditionally, so the file didn't compile on a machine without the CUDA
toolkit — including most of the actual edge targets (phones, laptops) and
any CI/dev machine without an NVIDIA GPU. `src/cpp/elora_allocator.cpp`
keeps the identical dependency-tree eviction semantics but selects the
allocation strategy at compile time via `USE_CUDA`, defaulting to plain
host allocation so it's usable and testable everywhere; `-DMSP_USE_CUDA=ON`
switches it back to the CUDA path unchanged.

### 6. Hardcoded telemetry (the watchdog could never observe a real event)

The v3.0 code read a fixed constant (`55.0`) as "the temperature," which
means the throttling/fallback logic could never be exercised by anything
real. `msp_telemetry_reader_fn` (C) and `ThermalMonitor`'s pluggable
`reader` (Python) replace the constant with an injectable source, so the
same logic can be driven by real hardware, a scripted test double, or
(eventually) a simulation.

### 7. "Must definitively fire every 10ms" / "<5ms fallback" as hard guarantees

Linux is not a real-time OS; no userspace polling loop can *guarantee* a
fixed period against scheduler preemption. `tests/daemon/test_watchdog.c`
measures actual jitter and fallback latency and checks them against a
generous sanity bound, instead of asserting an impossible guarantee. In
practice, the signal-based fallback here measures well under a
millisecond on this machine — comfortably inside the spec's aspirational
"<5ms" — but that's a measurement, not a promise.

### 8. Adapter dtype and device assumptions (Python layer)

The v3.0 async wrapper called `torch.cuda.*` unconditionally, which raises
on any CPU-only machine, and allocated adapter parameters as fixed
`float32` regardless of the base layer's dtype, silently breaking fp16/bf16
deployments — the exact memory-constrained case this project targets.
`StructuralPluginLayer` branches on `x.device.type` and matches the base
layer's dtype by default.

### 9. LoRA initialization and the routing gate

The v2 doc's Python reference (which is closer to upstream LoRA/PEFT
convention) uses `kaiming_uniform_(A, a=sqrt(5))` / `zeros_(B)` and a
dynamic gate bit `D` for a zero-compute bypass that doesn't free memory.
This repo's `StructuralPluginLayer` matches that initialization and adds
`routing_gate_enabled` for the same bypass behavior, distinct from
`AdapterManager.unload_adapter` (which does free memory).

## What's verified vs. what isn't

- **Verified in this environment:** the full Python package (23 unit
  tests, pytest), the C++ allocator and C daemon/integrity-check code
  (built and tested via CMake/ctest, and re-run clean under
  AddressSanitizer + UndefinedBehaviorSanitizer).
- **Verified on real GPU hardware, outside this environment (no GPU
  here):** the CUDA kernel (`src/cuda/`) — reviewed for correctness and
  syntax, then compiled with `nvcc` on a Colab T4 (2026-07-17) and
  numerically cross-checked against `StructuralPluginLayer`'s PyTorch
  autograd gradient via `tests/cuda/validate_gradient_kernel.py`
  (2026-07-20), both with thermal throttling on and off. See
  `docs/STATUS.md`'s "CUDA validation on Google Colab" section for the
  exact recipe and captured output. The TVM/ONNX cross-platform pipeline
  described in section 3.2 of the v2 doc is still not implemented at all.

For a status table and a diagram of which subsystems are (and are not)
actually wired together, see [`STATUS.md`](STATUS.md).

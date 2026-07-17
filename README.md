# MSP — Modular Structural Plugins

Hot-swappable, LoRA-style structural adapters for on-device model
adaptation: a frozen base model plus low-rank "plugins" that can be
loaded, activated, gated on/off, and evicted independently, under a
memory budget, with thermal-aware training and a watchdog-driven fallback
path.

This repo implements the core mechanics described in the `Project
Apex-Edge` design docs, with the real bugs and category errors those docs
contained fixed and documented.

**New here? Start with [`docs/STATUS.md`](docs/STATUS.md)** — it has a
full architecture map (what's built, what's wired together, what isn't)
and an honest "what's done / what's left" list, so you don't have to
reverse-engineer the repo to find out. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
lists exactly what was wrong with the original spec and how each issue
was fixed. [`docs/SECURITY.md`](docs/SECURITY.md) is an honest statement
of the threat model this repo does and does not cover.

## Layout

```
src/python/msp/    Reference implementation (PyTorch) — portable,
                    CPU-testable, the source of truth for behavior.
                    Also where msp_native (see below) gets built into.
src/cpp/            EloraAllocator (KV-cache allocator) + bindings.cpp
                    (pybind11 module wiring it into Python as msp_native).
src/cuda/           NVIDIA CUDA kernel for on-device gradient compute.
src/daemon/         C watchdog + SHA-256/Ed25519 integrity-check daemon.
tests/              Mirrors src/. Python via pytest; C/C++ via plain
                    executables run through ctest.
docs/               Architecture, status, and security notes.
```

## Quickstart — Python layer

```bash
pip install -r requirements.txt
PYTHONPATH=src/python python -m pytest tests/python -v
```

```python
import torch.nn as nn
from msp import StructuralPluginLayer, AdapterManager, ThermalMonitor

base = nn.Linear(4096, 4096)
layer = StructuralPluginLayer(base, rank=16, alpha=32.0)

# Zero-compute bypass (adapter stays resident, just inactive):
layer.routing_gate_enabled = False

# Full eviction (frees memory), via the manager:
mgr = AdapterManager()  # 512MB budget by default
mgr.load_adapter("legal-taxonomy-v1", {"proj": layer})
mgr.activate("legal-taxonomy-v1")
mgr.unload_adapter("legal-taxonomy-v1")

# Thermal-aware gradient gating during training:
monitor = ThermalMonitor(freeze_threshold_c=75.0, freeze_ratio=0.2)
layer.gate_gradients(monitor)
```

## Quickstart — C/C++/daemon layer

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This also builds `msp_native` (a pybind11 extension wiring `EloraAllocator`
into `AdapterManager`, dropped directly into `src/python/msp/`) as long as
`pybind11` is installed (`pip install pybind11`, or just `pip install -r
requirements.txt`). Without it, `AdapterManager` still works — it falls
back to pure-Python byte-count tracking. Check `AdapterManager().native_backed`
to see which mode you're in. Disable it explicitly with
`cmake -B build -DMSP_BUILD_PYTHON_BINDINGS=OFF`.

To build the KV-cache allocator against the real CUDA runtime (requires
the NVIDIA CUDA toolkit) instead of the portable host-allocation fallback:

```bash
cmake -B build -DMSP_USE_CUDA=ON
```

## What's verified vs. what isn't

- **Verified in this environment:** the full Python package (pytest), the
  C++ allocator, the pybind11 binding wiring it into Python, and the C
  daemon/integrity-check code (Ed25519 signing included) — built and
  tested via CMake/ctest, and re-run clean under AddressSanitizer +
  UndefinedBehaviorSanitizer.
- **Not verified here:** `src/cuda/fused_msp_backward_kernel.cu` — this
  development environment has no NVIDIA GPU or CUDA toolkit. The kernel
  has been reviewed for correctness and syntax but not compiled or
  executed. See the caveat comment at the top of that file for the exact
  validation recipe before trusting it in production. The ONNX → Apache
  TVM/LLVM cross-platform compilation pipeline described in the v2 design
  doc is not implemented at all — it needs the TVM toolchain and vendor
  SDKs (Xcode, Qualcomm Hexagon SDK) that aren't available here.

See [`docs/STATUS.md`](docs/STATUS.md) for the full, itemized breakdown of
what's done vs. left to do, and how the pieces (do and don't) connect.

## License

Not yet chosen — add a `LICENSE` file with whichever license fits your
plans for this repo (MIT and Apache-2.0 are the common defaults for a
project like this).

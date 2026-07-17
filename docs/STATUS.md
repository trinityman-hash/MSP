# MSP Project Status & Architecture Map

Read this first if you're new to the repo. It answers two questions:
"what actually works right now" and "how do the pieces fit together."
For *why* things were built this way (bugs found and fixed vs. the
original spec), see [`ARCHITECTURE.md`](ARCHITECTURE.md). For the threat
model, see [`SECURITY.md`](SECURITY.md).

## TL;DR status table

| Component | Status | Tested how | Notes |
|---|---|---|---|
| `StructuralPluginLayer` (Python) | Done | 12 pytest cases | Core LoRA adapter, gate, device/dtype-aware |
| `AdapterManager` (Python) | Done | 12 pytest cases | Memory-budget enforcement, hot-swap, now backed by real native allocation |
| `ThermalMonitor` (Python) | Done | 5 pytest cases | Pluggable reader interface |
| `LinuxThermalZoneReader` (Python) | Done | 12 pytest cases | Real `/sys/class/thermal` reader; tested against a simulated sysfs tree (no real thermal zones in this dev container) |
| Adapter persistence (`msp.persistence`) | Done | 8 pytest cases | safetensors save/load, round-trip and multi-layer tested |
| `EloraAllocator` (C++) | Done | 6 cases via ctest | Portable by default; CUDA path unverified |
| `msp_native` (pybind11 binding) | Done | 5 pytest cases + ASan smoke test | Wires EloraAllocator into AdapterManager for real |
| `sandbox_watchdog` (C) | Done | 2 scenarios via ctest | Signal-based fallback; sub-ms measured latency; telemetry reader still test-double only on the C side |
| `integrity_check` (C, SHA-256 + Ed25519) | Done | Known-answer + tamper + signature round-trip tests | Signing primitive done; trust-root distribution still a deployment decision |
| CUDA kernel | Written, **not run** | Manual code review only | No GPU in this environment — see "CUDA validation on Google Colab" below |
| ONNX -> TVM/LLVM pipeline | **Not started** | — | Needed for the real cross-platform story (see below) |
| B2B marketplace / SDK / registry | **Not started** | — | This is Phase 3 in the v2 doc; nothing here yet |
| CI actually running on GitHub | **Unverified** | — | Workflow file is written and passes locally; not yet confirmed green in GitHub Actions |
| License | **Not chosen** | — | No `LICENSE` file yet |

## Full architecture map

How the pieces conceptually relate. Solid arrows are real, working
connections. Dashed arrows are described in the design docs but **not
implemented** — this is the most important thing for a new contributor to
notice, because it's easy to assume a diagram like this describes a
working end-to-end system, and it doesn't yet.

```mermaid
flowchart TB
    subgraph Python["src/python/msp/  (fully working, CPU-testable)"]
        PL["StructuralPluginLayer\n(LoRA adapter + gate)"]
        AM["AdapterManager\n(memory budget + hot-swap)"]
        TM["ThermalMonitor\n(pluggable telemetry)"]
        LTZ["LinuxThermalZoneReader\n(real sysfs telemetry)"]
        PERSIST["persistence.py\n(safetensors save/load)"]
        AM -->|owns 1..N| PL
        TM -.->|drives, via gate_gradients| PL
        LTZ -.->|"can be passed as\nThermalMonitor(reader=...)"| TM
        PERSIST -.->|reconstructs| PL
    end

    subgraph Bind["src/cpp/bindings.cpp  (pybind11, WIRED)"]
        NATIVE["msp_native.EloraAllocator\n(Python-callable)"]
    end

    subgraph CPP["src/cpp/  (fully working, CPU-testable)"]
        EA["EloraAllocator\n(KV-cache, dependency-tree eviction)"]
    end

    subgraph C["src/daemon/  (fully working, CPU-testable)"]
        WD["sandbox_watchdog\n(thermal/memory/security watchdog)"]
        IC["integrity_check\n(SHA-256 + Ed25519 signing)"]
    end

    subgraph CUDA["src/cuda/  (written, UNVERIFIED)"]
        KERNEL["fused_msp_backward_kernel\n(thermal-gated gradient compute)"]
    end

    AM ==>|"WIRED: real allocate/evict\ncalls on every load/unload"| NATIVE
    NATIVE ==>|pybind11| EA
    TM -.->|"NOT WIRED: no bindings call\nsandbox_watchdog from Python"| WD
    PL -.->|"NOT WIRED: kernel is not called\nfrom the Python training loop"| KERNEL
    IC -.->|"NOT WIRED: no adapter-loading code\npath calls this yet"| PL

    subgraph Missing["Not started at all"]
        TVM["ONNX -> Apache TVM/LLVM\ncross-platform compiler"]
        SDK["B2B marketplace SDK\n+ adapter registry"]
    end
```

In words: `AdapterManager.load_adapter()` / `unload_adapter()` now perform
**real** allocation and eviction through `EloraAllocator`, via a pybind11
binding (`src/cpp/bindings.cpp` → `msp_native`), verified end-to-end (byte
counts match between the Python and native sides, eviction actually frees
native memory, budget rejection never touches the native allocator). This
is optional and gracefully degrades: if the extension hasn't been built
(no C++ toolchain, or simply not built yet), `AdapterManager` falls back
to its original pure-Python byte-tracking behavior automatically — check
`AdapterManager.native_backed` to see which mode you're in.

Still not wired: `ThermalMonitor` doesn't yet read from
`sandbox_watchdog`'s telemetry struct (they remain two independent
pluggable-reader systems, one Python one C, not sharing a data source),
and the CUDA kernel is not called from the Python training loop. See
"What's left to do" below.

## What's done

- **Python reference implementation** (`src/python/msp/`): the LoRA-style
  adapter with the dynamic gate, the memory-budgeted hot-swap manager, and
  pluggable thermal-aware gradient gating. 29 passing pytest cases,
  including regression tests for every bug fixed relative to the original
  spec (see `ARCHITECTURE.md` for the list).
- **C++ KV-cache allocator** (`src/cpp/elora_allocator.*`): adapter-scoped
  dependency-tree eviction, portable by default with an opt-in CUDA build
  path. 6 test cases, clean under AddressSanitizer + UndefinedBehaviorSanitizer.
- **Python <-> C++ binding** (`src/cpp/bindings.cpp` → `msp_native`):
  pybind11 extension wiring `EloraAllocator` into `AdapterManager`, so
  `load_adapter`/`unload_adapter` perform real allocation/eviction, not
  just Python-side byte counting. Optional and auto-detected — falls back
  to the original pure-Python behavior if the extension isn't built.
  5 dedicated pytest cases plus a standalone ASan/UBSan smoke test
  (isolated from PyTorch's own import-time allocations, which show up as
  unrelated "leaks" under ASan if you test through the full `msp` package
  import — see the binding's own verification notes for how to reproduce
  the isolated check).
- **C watchdog + integrity daemon** (`src/daemon/`): signal-based
  (not cross-thread-`longjmp`-based) fallback mechanism, measured
  sub-millisecond in this environment; SHA-256 integrity hashing with
  known-answer tests, **plus real Ed25519 signing/verification**
  (`msp_ed25519_sign` / `msp_verify_signature`) with round-trip and
  tamper-detection tests. Also clean under sanitizers.
- **Adapter persistence** (`src/python/msp/persistence.py`): save/load a
  set of `StructuralPluginLayer` weights to a single `.safetensors` file
  (rank, alpha, and the routing gate state are preserved as metadata
  alongside the tensors). Chosen over pickle-based `torch.save` because
  safetensors can't execute code on load -- relevant since this format is
  meant to eventually carry adapters from third-party publishers (the B2B
  marketplace use case). 8 pytest cases, including a check that a loaded
  layer produces bit-for-bit identical output to the original. Does not
  itself verify integrity/authenticity of a loaded file -- pair with
  `integrity_check.c`'s functions (via a future Python binding) for
  untrusted sources.
- **Real Linux thermal telemetry reader** (`LinuxThermalZoneReader` in
  `thermal.py`): reads real `/sys/class/thermal/thermal_zone*/temp`
  values (converting the kernel's millidegree-Celsius units), with
  zone-type filtering (e.g. only "cpu" zones) and a choice of max/mean
  aggregation across zones. This container has no real thermal zones to
  read, so it's tested against a simulated sysfs directory tree built in
  the test itself (12 pytest cases) -- the parsing/aggregation/error-path
  logic is fully exercised; what's NOT exercised is reading actual
  hardware, which needs to happen on a real Linux machine. The C side
  (`msp_telemetry_reader_fn` in `sandbox_watchdog.h`) still only has a
  scripted test double, not a real reader -- see "What's left to do".
- **Build system**: CMake for the C/C++/daemon/bindings layer (with
  `-fPIC` enabled globally so the static libs link cleanly into the
  pybind11 shared module — an actual bug caught and fixed while wiring
  this up, see "Bugs found while continuing this work" below),
  `pyproject.toml` for the Python package, a GitHub Actions workflow
  covering both (CPU-only — the CUDA path is explicitly out of scope for
  CI, since no GPU runner is configured).
- **Documentation**: this file, `ARCHITECTURE.md` (what was fixed and
  why), `SECURITY.md` (honest threat-model statement).

## Bugs found while continuing this work

Two more real bugs surfaced while building on top of the initial
implementation, beyond the ones in `ARCHITECTURE.md`:

- **Missing `-fPIC`**: linking `elora_allocator` (a static lib, compiled
  without position-independent code by CMake's default) into
  `msp_native.so` (a shared object) failed at link time with
  `relocation R_X86_64_PC32 ... can not be used when making a shared
  object`. Fixed by setting `CMAKE_POSITION_INDEPENDENT_CODE ON` globally
  in `CMakeLists.txt`.
- **Wrong import path for the compiled extension**: the pybind11 module
  is built directly into `src/python/msp/` (correct — that's where a
  compiled extension belongs inside a proper Python package), but the
  first version of `adapter_manager.py` used `import msp_native`
  (absolute/top-level) instead of `from . import msp_native` (relative),
  so it silently fell back to pure-Python mode with no error, only
  discovered by explicitly checking `AdapterManager.native_backed` after
  the "successful" build. Fixed, and now covered by
  `test_native_backend_availability_is_reported_honestly`.

## What's left to do

Roughly in the order a next contributor would probably want to tackle
them:

1. **Validate the CUDA kernel on real hardware.** It's written and
   reviewed but never compiled or run in this environment (no GPU). See
   "CUDA validation on Google Colab" below for the exact process to do
   this on Colab's free-tier T4 GPU.
2. **Real C-side telemetry reader.** The Python side now has
   `LinuxThermalZoneReader` (real sysfs reads); the C side
   (`msp_telemetry_reader_fn` in `sandbox_watchdog.h`) still only has the
   scripted test double from `tests/daemon/test_watchdog.c`. Either port
   the same sysfs-reading logic to C, or (probably better) bind
   `LinuxThermalZoneReader` through pybind11 so both languages share one
   implementation instead of maintaining two.
3. **Wire `ThermalMonitor` to `sandbox_watchdog`'s real telemetry.** The
   Python↔C++ allocator binding is done; the thermal side is still two
   independent pluggable-reader systems (Python's `ThermalMonitor.reader`
   and C's `msp_telemetry_reader_fn`) that don't share a data source.
4. **End-to-end training example.** Everything here is unit-tested in
   isolation; there's no example wiring a `StructuralPluginLayer` into an
   actual multi-layer transformer block and running a real training loop
   against it.
5. **Trust-root decision for signature verification.** The cryptographic
   primitive is done (`msp_verify_signature`, Ed25519, tested) — what's
   left is a deployment decision about where a verifier gets a public key
   it should trust (embedded key vs. certificate chain vs. attestation
   service) and how revocation works. See `SECURITY.md`.
6. **The ONNX -> Apache TVM/LLVM cross-platform pipeline.** This is the
   architecturally-sound way (per the v2 design doc) to actually deploy
   across Apple AMX / Qualcomm Hexagon / etc. Nothing toward this exists
   yet; it needs the TVM toolchain and vendor SDKs.
7. **B2B marketplace SDK / adapter registry.** Phase 3 in the v2 doc's
   roadmap. Not started — arguably shouldn't be, until steps 1-4 above
   give you something worth registering.
8. **Housekeeping:** pick a `LICENSE`, confirm the GitHub Actions workflow
   is actually green on real GitHub infrastructure (it's only been
   validated by running the equivalent commands locally), and decide on a
   versioning/release process once there's a first real consumer of this
   package.

## CUDA validation on Google Colab

`src/cuda/fused_msp_backward_kernel.cu` has never been compiled or
executed — this development environment has no NVIDIA GPU. Colab's free
tier provides one (a T4), which is enough to validate it. Exact process:

1. Go to **colab.research.google.com** → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. In the first cell, clone the repo and confirm the GPU is visible:
   ```python
   !git clone https://github.com/trinityman-hash/MSP.git
   %cd MSP
   !nvidia-smi
   ```
   `nvidia-smi` should print a T4 in its table. If it doesn't, the
   runtime type didn't actually switch to GPU — repeat step 2.
4. Compile the kernel (Colab images ship `nvcc` preinstalled):
   ```python
   !nvcc -O3 -arch=sm_75 --compiler-options -Wall \
       -c src/cuda/fused_msp_backward_kernel.cu \
       -o fused_msp_backward_kernel.o
   !echo "compiled: $?"
   ```
   (T4's compute capability is 7.5, hence `sm_75` — different from the
   `sm_80` example in the file's own header comment, which assumed an
   A100-class card.)
5. **This only proves it compiles, not that it's correct.** The real
   check is cross-validating the kernel's output against
   `StructuralPluginLayer`'s own PyTorch autograd gradient, per the
   validation recipe already written into the top of
   `fused_msp_backward_kernel.cu`. That requires a small host (Python)
   wrapper that loads the compiled kernel via `ctypes`/`cupy`/a tiny
   pybind11 module, feeds it the same fixed input PyTorch computed a
   gradient for, and diffs the two results numerically (`torch.allclose`,
   not exact equality — floating point). That wrapper does not exist yet;
   writing it is the concrete next step, and worth doing on Colab
   directly since it needs the GPU to run either way.

Since this needs to happen from a phone: Colab's notebook UI works in a
mobile browser exactly as the steps above describe — each `!command` goes
in its own cell, run with the ▶ button, and output/errors print directly
below the cell.

## How to verify this status yourself

```bash
# Install everything, including pybind11 for the native binding
pip install -r requirements.txt

# C/C++/daemon/bindings (builds msp_native.so into src/python/msp/ automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure

# Python (49 tests -- 5 of these specifically exercise the native binding
# built just above; they're skipped, not failed, if you skip that step)
PYTHONPATH=src/python python -m pytest tests/python -v

# Same C/C++ tests, under AddressSanitizer + UndefinedBehaviorSanitizer
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMSP_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -g -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

Nothing in `src/cuda/` is part of these commands — there is currently no
automated way to validate it without NVIDIA GPU hardware.

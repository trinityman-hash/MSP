# MSP Security Notes

This document exists because the original spec's naming ("Ring 3 Security
Sandbox," "RSA-4096 hash validation") implied stronger guarantees than the
described mechanisms actually provide. This is an honest statement of what
this repo does and does not protect against today.

## Threat model this repo actually addresses

1. **Runaway on-device training** (thermal or memory runaway during local
   gradient descent): addressed by `ThermalMonitor` (Python) /
   `sandbox_watchdog.c` (C) — a pluggable telemetry reader plus a
   throttle-then-fallback policy. This is a **safety** mechanism (protects
   the device and the training job from a bug or a bad input), not a
   **security** mechanism (does not protect against an adversarial
   adapter trying to escape confinement).

2. **Tampered/corrupted adapter payloads in transit or at rest**:
   addressed by `integrity_check.c`'s SHA-256 hashing —
   `msp_verify_integrity()` detects if bytes were altered from a
   previously-published digest. This is **integrity**, not
   **authenticity**: it tells you the bytes match a hash you already
   trusted, not that the hash itself came from a legitimate publisher.

3. **Adapter memory budget exhaustion**: addressed by `AdapterManager`
   (Python) and `EloraAllocator`'s dependency-tree eviction (C++) — fails
   fast at load time rather than letting the OS's own OOM killer decide.

## Threat model this repo does NOT address (and what would be needed)

### Authenticity of third-party adapters (the B2B marketplace use case)

If adapters are meant to be purchased/licensed from third parties (as the
v2 doc's "AI Parts marketplace" describes), a device needs to verify an
adapter was published by whoever it claims to be from, not just that its
bytes weren't corrupted. That needs:

- A signature scheme (Ed25519 recommended over RSA for this: smaller
  signatures, faster verification, less error-prone implementation) over
  the SHA-256 digest already computed by `integrity_check.c`.
- A trust root: either an embedded public key per publisher, a
  certificate chain, or a remote attestation/revocation service.
- A decision about revocation: what happens to an adapter already resident
  on-device if its signing key is later revoked?

`msp_verify_signature_STUB()` in `integrity_check.h` is a deliberately
unimplemented placeholder with the right function shape for this — it
returns `false` unconditionally so nothing accidentally treats an
unimplemented check as a passed one. None of the trust-root decisions
above are made in this repo; they're deployment-specific.

### Isolation of untrusted adapter code/weights

`msp_allocate_isolated_region()` maps memory without `PROT_EXEC`, which is
a real, useful property: data placed there can't be directly executed as
machine code via that mapping. That is **not** the same as running
untrusted code (or untrusted weight-triggered computation) in a
confined environment. If adapters can influence anything beyond pure
tensor math on already-validated shapes — custom ops, arbitrary file
access, network access — real isolation needs one of:

- A separate OS process with a restricted syscall filter (seccomp-bpf on
  Linux; sandbox profiles on iOS/Android), so a compromised or malicious
  adapter can't reach the filesystem, network, or other processes even if
  it achieves arbitrary code execution somehow.
- A memory-safe interpreted/JIT runtime for adapter-defined computation
  (e.g. a WASM runtime like wasmtime/wasmer), so there's no native code
  execution surface at all.

Neither is implemented here. For the current scope (adapters are just
weight tensors consumed by a fixed, trusted set of matrix operations —
matmuls with the frozen base layer), the risk surface is much smaller than
"arbitrary adapter code execution," but that assumption should be
re-examined if the SDK ever lets adapters define custom ops.

### Side-channel resistance

`msp_digest_equal()` does a constant-time comparison so hash-checking
doesn't leak timing information about how many leading bytes matched. That
is the only side-channel mitigation in this repo. It does not address
power-analysis, cache-timing attacks on the matrix operations themselves,
or anything at the hardware level — those would need to be evaluated
against the actual target hardware's threat model, which is outside the
scope of a software-only review.

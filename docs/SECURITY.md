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
bytes weren't corrupted.

**The cryptographic primitive for this is now implemented**:
`msp_ed25519_sign()` / `msp_verify_signature()` in `integrity_check.c`
sign and verify Ed25519 signatures over the SHA-256 digest already
computed by `integrity_check.c`'s integrity half. Ed25519 was chosen over
RSA for the reasons above (fixed-size keys/signatures, no padding-scheme
footguns, constant-time by construction). This has known-answer,
round-trip, and tamper-detection tests in `tests/daemon/test_integrity_check.c`.

**What's still a deployment decision, not implemented here:**

- A trust root: how a verifier obtains and trusts a specific public key
  in the first place -- an embedded key per publisher, a certificate
  chain, or a remote attestation/revocation service.
- Key management for the *signing* side: `msp_ed25519_generate_keypair()`
  exists for tests/local development only; a real publisher's signing key
  needs proper generation, storage, and access-control practices this
  repo has no opinion on.
- Revocation: what happens to an adapter already resident on-device if
  its signing key is later revoked?

In short: the "can I verify this specific signature against this specific
key" question is answered. The "should I trust this key" question is not,
and depends on decisions (PKI vs. embedded keys vs. attestation) that
belong to whoever operates the marketplace.

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

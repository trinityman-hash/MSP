// integrity_check.h
//
// Replaces the spec's placeholder comment
//   "[Cryptographic RSA-4096 Hash Validation Logic goes here]"
// which conflated two different primitives:
//
//   - INTEGRITY  ("has this data been corrupted or tampered with?")
//     is answered by a cryptographic HASH, e.g. SHA-256. RSA is not a
//     hash function and has no role here on its own.
//
//   - AUTHENTICITY ("did this data really come from a trusted publisher,
//     e.g. the B2B adapter marketplace?") is answered by a SIGNATURE
//     scheme applied *to* a hash of the data.
//
// This module implements BOTH halves concretely via OpenSSL:
//   - SHA-256 for integrity (msp_sha256 / msp_verify_integrity).
//   - Ed25519 for authenticity (msp_ed25519_sign / msp_verify_signature),
//     chosen over RSA per the tradeoffs in docs/SECURITY.md: fixed
//     64-byte signatures, fixed 32-byte keys, and an implementation with
//     far fewer footguns than RSA (no padding-scheme choices to get
//     wrong, constant-time by construction).
//
// WHAT THIS DOES NOT DECIDE: where a verifier gets a public key it should
// actually trust (an embedded key vs. a certificate chain vs. a remote
// attestation/revocation service), or how key rotation/revocation works.
// Those are deployment-specific trust-root decisions -- see
// docs/SECURITY.md. msp_verify_signature() below takes the public key as
// a parameter precisely so the caller supplies it however their trust
// model dictates; this module only implements the cryptographic
// operation once you have one.

#ifndef MSP_INTEGRITY_CHECK_H
#define MSP_INTEGRITY_CHECK_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MSP_SHA256_DIGEST_LEN 32

// Computes SHA-256(data[0..len)) into out_digest (must be
// MSP_SHA256_DIGEST_LEN bytes). Returns true on success.
bool msp_sha256(const uint8_t* data, size_t len,
                 uint8_t out_digest[MSP_SHA256_DIGEST_LEN]);

// Constant-time comparison of two digests, to avoid timing side-channels
// when checking against an expected hash (the original spec's stated
// concern was "side-channel probes" from untrusted plugins -- a naive
// memcmp of a secret-derived value is itself a small side channel).
bool msp_digest_equal(const uint8_t a[MSP_SHA256_DIGEST_LEN],
                       const uint8_t b[MSP_SHA256_DIGEST_LEN]);

// Verify that `data` matches an expected, previously-published digest.
bool msp_verify_integrity(const uint8_t* data, size_t len,
                           const uint8_t expected_digest[MSP_SHA256_DIGEST_LEN]);

// ---------------------------------------------------------------------
// Authenticity (Ed25519). Fixed-size raw keys/signatures -- no ASN.1, no
// padding scheme to choose, matching Ed25519's usual (and recommended)
// raw-bytes usage.
// ---------------------------------------------------------------------

#define MSP_ED25519_PUBKEY_LEN 32
#define MSP_ED25519_PRIVKEY_LEN 32
#define MSP_ED25519_SIGNATURE_LEN 64

// Generates a new Ed25519 keypair. Intended for tests and local
// development -- a real deployment's signing key belongs to whoever
// publishes adapters (the B2B marketplace side), generated and stored
// with proper key-management practices well outside what a header-only
// helper like this should be trusted for.
bool msp_ed25519_generate_keypair(uint8_t out_public_key[MSP_ED25519_PUBKEY_LEN],
                                   uint8_t out_private_key[MSP_ED25519_PRIVKEY_LEN]);

// Signs `digest` (typically the output of msp_sha256 over an adapter's
// bytes) with the given private key.
bool msp_ed25519_sign(const uint8_t private_key[MSP_ED25519_PRIVKEY_LEN],
                       const uint8_t digest[MSP_SHA256_DIGEST_LEN],
                       uint8_t out_signature[MSP_ED25519_SIGNATURE_LEN]);

// Verifies that `signature` is a valid Ed25519 signature over `digest`
// under `public_key`. This answers "did the holder of the private key
// matching this specific public key sign this exact digest" -- it does
// NOT answer "should I trust this public key in the first place." That
// second question is the trust-root decision documented in
// docs/SECURITY.md and is the caller's responsibility.
bool msp_verify_signature(const uint8_t digest[MSP_SHA256_DIGEST_LEN],
                           const uint8_t signature[MSP_ED25519_SIGNATURE_LEN],
                           const uint8_t public_key[MSP_ED25519_PUBKEY_LEN]);

#ifdef __cplusplus
}
#endif

#endif  // MSP_INTEGRITY_CHECK_H

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
//     scheme (RSA-PSS or, preferably, Ed25519) applied *to* a hash of the
//     data. This is a separate, optional step -- msp_verify_signature()
//     is provided as a stub with the correct shape, since wiring up a
//     real signing/trust-root workflow is a deployment-specific decision
//     (key management, revocation, etc.) outside the scope of this fix.
//
// This module implements the hash/integrity half concretely (via
// OpenSSL's EVP SHA-256), and documents the signature half so it's built
// correctly if/when a trust root is available.

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

// NOT IMPLEMENTED: signature verification (authenticity). Wiring this up
// requires a decision about the trust root (embedded public key vs. a
// certificate chain vs. a remote attestation service) that belongs to
// deployment configuration, not this library. Left as a documented stub
// so the correct primitive (a signature over the SHA-256 digest, verified
// with e.g. RSA-PSS or Ed25519) is the one that gets filled in here,
// rather than an ad hoc "RSA hash."
bool msp_verify_signature_STUB(const uint8_t digest[MSP_SHA256_DIGEST_LEN],
                                const uint8_t* signature, size_t sig_len,
                                const uint8_t* public_key, size_t key_len);

#ifdef __cplusplus
}
#endif

#endif  // MSP_INTEGRITY_CHECK_H

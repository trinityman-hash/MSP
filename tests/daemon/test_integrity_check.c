// Standalone test for integrity_check.c. No external test framework.

#include "integrity_check.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAILED: %s (line %d)\n", #cond, __LINE__);    \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main(void) {
    // Known-answer test: SHA-256("") is a well-known constant.
    const uint8_t sha256_empty[MSP_SHA256_DIGEST_LEN] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
    };
    uint8_t digest[MSP_SHA256_DIGEST_LEN];
    CHECK(msp_sha256((const uint8_t*)"", 0, digest));
    CHECK(msp_digest_equal(digest, sha256_empty));

    // Known-answer test: SHA-256("abc")
    const uint8_t sha256_abc[MSP_SHA256_DIGEST_LEN] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    CHECK(msp_sha256((const uint8_t*)"abc", 3, digest));
    CHECK(msp_digest_equal(digest, sha256_abc));

    // Tamper detection: flipping one byte of input changes the digest.
    uint8_t d1[MSP_SHA256_DIGEST_LEN], d2[MSP_SHA256_DIGEST_LEN];
    const char* original = "structural-plugin-payload-v1";
    char tampered[64];
    strcpy(tampered, original);
    tampered[0] ^= 0x01;

    CHECK(msp_sha256((const uint8_t*)original, strlen(original), d1));
    CHECK(msp_sha256((const uint8_t*)tampered, strlen(tampered), d2));
    CHECK(!msp_digest_equal(d1, d2));

    // verify_integrity end-to-end
    CHECK(msp_verify_integrity((const uint8_t*)original, strlen(original), d1));
    CHECK(!msp_verify_integrity((const uint8_t*)tampered, strlen(tampered), d1));

    // --- Ed25519 signature (authenticity) tests ---

    uint8_t pub_key[MSP_ED25519_PUBKEY_LEN];
    uint8_t priv_key[MSP_ED25519_PRIVKEY_LEN];
    CHECK(msp_ed25519_generate_keypair(pub_key, priv_key));

    uint8_t signature[MSP_ED25519_SIGNATURE_LEN];
    CHECK(msp_ed25519_sign(priv_key, d1, signature));

    // Valid signature over the digest it was actually signed for, with
    // the matching public key, must verify.
    CHECK(msp_verify_signature(d1, signature, pub_key));

    // A signature must NOT verify against a different digest (someone
    // swapped the payload after signing).
    CHECK(!msp_verify_signature(d2, signature, pub_key));

    // A signature must NOT verify against a different (unrelated) public
    // key (someone claims a different signer).
    uint8_t other_pub[MSP_ED25519_PUBKEY_LEN];
    uint8_t other_priv[MSP_ED25519_PRIVKEY_LEN];
    CHECK(msp_ed25519_generate_keypair(other_pub, other_priv));
    CHECK(!msp_verify_signature(d1, signature, other_pub));

    // Flipping a single bit of a valid signature must invalidate it.
    uint8_t corrupted_signature[MSP_ED25519_SIGNATURE_LEN];
    memcpy(corrupted_signature, signature, MSP_ED25519_SIGNATURE_LEN);
    corrupted_signature[0] ^= 0x01;
    CHECK(!msp_verify_signature(d1, corrupted_signature, pub_key));

    // Two keypairs generated back-to-back must not be identical (sanity
    // check that keygen is actually randomized, not returning a fixed
    // "test" key).
    CHECK(memcmp(pub_key, other_pub, MSP_ED25519_PUBKEY_LEN) != 0);

    printf("All integrity_check tests passed.\n");
    return 0;
}

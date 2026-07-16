// integrity_check.c
// See integrity_check.h for rationale.

#include "integrity_check.h"

#include <openssl/evp.h>

bool msp_sha256(const uint8_t* data, size_t len,
                 uint8_t out_digest[MSP_SHA256_DIGEST_LEN]) {
    if (data == NULL && len != 0) {
        return false;
    }
    unsigned int out_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == NULL) {
        return false;
    }
    bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1 &&
              EVP_DigestUpdate(ctx, data, len) == 1 &&
              EVP_DigestFinal_ex(ctx, out_digest, &out_len) == 1 &&
              out_len == MSP_SHA256_DIGEST_LEN;
    EVP_MD_CTX_free(ctx);
    return ok;
}

bool msp_digest_equal(const uint8_t a[MSP_SHA256_DIGEST_LEN],
                       const uint8_t b[MSP_SHA256_DIGEST_LEN]) {
    // Constant-time compare: always touch every byte regardless of where
    // a mismatch occurs, so timing doesn't leak how many leading bytes
    // matched.
    uint8_t diff = 0;
    for (int i = 0; i < MSP_SHA256_DIGEST_LEN; i++) {
        diff |= (uint8_t)(a[i] ^ b[i]);
    }
    return diff == 0;
}

bool msp_verify_integrity(const uint8_t* data, size_t len,
                           const uint8_t expected_digest[MSP_SHA256_DIGEST_LEN]) {
    uint8_t actual[MSP_SHA256_DIGEST_LEN];
    if (!msp_sha256(data, len, actual)) {
        return false;
    }
    return msp_digest_equal(actual, expected_digest);
}

bool msp_verify_signature_STUB(const uint8_t digest[MSP_SHA256_DIGEST_LEN],
                                const uint8_t* signature, size_t sig_len,
                                const uint8_t* public_key, size_t key_len) {
    (void)digest;
    (void)signature;
    (void)sig_len;
    (void)public_key;
    (void)key_len;
    // Deliberately unimplemented -- see header. Returning false (rather
    // than true) is the safe default: callers must not treat an
    // unimplemented check as a passed check.
    return false;
}

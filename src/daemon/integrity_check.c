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

bool msp_ed25519_generate_keypair(uint8_t out_public_key[MSP_ED25519_PUBKEY_LEN],
                                   uint8_t out_private_key[MSP_ED25519_PRIVKEY_LEN]) {
    bool ok = false;
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, NULL);
    EVP_PKEY* pkey = NULL;

    if (ctx == NULL) {
        goto done;
    }
    if (EVP_PKEY_keygen_init(ctx) != 1) {
        goto done;
    }
    if (EVP_PKEY_keygen(ctx, &pkey) != 1) {
        goto done;
    }

    {
        size_t pub_len = MSP_ED25519_PUBKEY_LEN;
        size_t priv_len = MSP_ED25519_PRIVKEY_LEN;
        if (EVP_PKEY_get_raw_public_key(pkey, out_public_key, &pub_len) != 1 ||
            pub_len != MSP_ED25519_PUBKEY_LEN) {
            goto done;
        }
        if (EVP_PKEY_get_raw_private_key(pkey, out_private_key, &priv_len) != 1 ||
            priv_len != MSP_ED25519_PRIVKEY_LEN) {
            goto done;
        }
    }
    ok = true;

done:
    if (pkey != NULL) EVP_PKEY_free(pkey);
    if (ctx != NULL) EVP_PKEY_CTX_free(ctx);
    return ok;
}

bool msp_ed25519_sign(const uint8_t private_key[MSP_ED25519_PRIVKEY_LEN],
                       const uint8_t digest[MSP_SHA256_DIGEST_LEN],
                       uint8_t out_signature[MSP_ED25519_SIGNATURE_LEN]) {
    bool ok = false;
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_ED25519, NULL, private_key, MSP_ED25519_PRIVKEY_LEN);
    EVP_MD_CTX* ctx = NULL;

    if (pkey == NULL) {
        goto done;
    }
    ctx = EVP_MD_CTX_new();
    if (ctx == NULL) {
        goto done;
    }
    // Ed25519 is a "one-shot" scheme in OpenSSL's EVP API: no
    // EVP_DigestSignUpdate calls, no message-digest type (Ed25519 does
    // its own internal SHA-512 over whatever buffer you hand it here --
    // in this module that buffer is our own SHA-256 digest, not the raw
    // adapter bytes; see the header comment for why that's fine).
    if (EVP_DigestSignInit(ctx, NULL, NULL, NULL, pkey) != 1) {
        goto done;
    }
    {
        size_t sig_len = MSP_ED25519_SIGNATURE_LEN;
        if (EVP_DigestSign(ctx, out_signature, &sig_len, digest, MSP_SHA256_DIGEST_LEN) != 1 ||
            sig_len != MSP_ED25519_SIGNATURE_LEN) {
            goto done;
        }
    }
    ok = true;

done:
    if (ctx != NULL) EVP_MD_CTX_free(ctx);
    if (pkey != NULL) EVP_PKEY_free(pkey);
    return ok;
}

bool msp_verify_signature(const uint8_t digest[MSP_SHA256_DIGEST_LEN],
                           const uint8_t signature[MSP_ED25519_SIGNATURE_LEN],
                           const uint8_t public_key[MSP_ED25519_PUBKEY_LEN]) {
    bool ok = false;
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519, NULL, public_key, MSP_ED25519_PUBKEY_LEN);
    EVP_MD_CTX* ctx = NULL;

    if (pkey == NULL) {
        goto done;
    }
    ctx = EVP_MD_CTX_new();
    if (ctx == NULL) {
        goto done;
    }
    if (EVP_DigestVerifyInit(ctx, NULL, NULL, NULL, pkey) != 1) {
        goto done;
    }
    // EVP_DigestVerify returns 1 for a valid signature, 0 for an invalid
    // one, and <0 for an actual error -- all non-1 results must be
    // treated as "not verified," not just genuine errors, so a malformed
    // signature can never be mistaken for a passing check.
    ok = (EVP_DigestVerify(ctx, signature, MSP_ED25519_SIGNATURE_LEN,
                            digest, MSP_SHA256_DIGEST_LEN) == 1);

done:
    if (ctx != NULL) EVP_MD_CTX_free(ctx);
    if (pkey != NULL) EVP_PKEY_free(pkey);
    return ok;
}

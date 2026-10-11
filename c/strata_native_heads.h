/* Native independent proposals, compiled in a separate non-fast-math object.
 * CSNH v1 stores true CSQ1 signed INT4/8/16 weights, decoded ONCE to immutable
 * F32. Prediction uses F32 activations and scalar/AVX2 double accumulation;
 * this is CPU proposal computation, not native integer accelerator arithmetic.
 * Every head reads the same hidden h_t; outputs cover offsets 2..head_count+1.
 * The target's offset-1 selection precedes these proposals and must verify them.
 */
#ifndef COLIBRI_STRATA_NATIVE_HEADS_H
#define COLIBRI_STRATA_NATIVE_HEADS_H
#include <stddef.h>
#include <stdint.h>

#define SN_MAX_HEADS 8u
#define SN_MAX_RANK 4096u
#define SN_MAX_FILE_BYTES ((size_t)512 * 1024 * 1024)
#define SN_MAX_DECODED_BYTES SN_MAX_FILE_BYTES
#define SN_HEADER_BYTES 96u

typedef struct sn_bank sn_bank;
typedef enum {
    SN_OK = 0, SN_INVALID, SN_IO, SN_IDENTITY, SN_DIMENSIONS,
    SN_LIMIT, SN_NONFINITE, SN_OVERFLOW, SN_NOMEM
} sn_status;

typedef struct {
    size_t hidden_dim, vocab_size, head_count, decoded_bytes;
    unsigned source_bits;
    const char *kernel;          /* borrowed static string: scalar or avx2 */
} sn_info;

/* Fingerprint root config.json/tokenizer.json (required), qwen36_meta.json
 * (optional), and every root *.safetensors file (>=1 required), sorted by UTF8
 * filename bytes, with tagged runtime-precision, names, lengths and contents.
 * Directories are not traversed. The root path itself is NOT hashed.
 * Files must stay unchanged throughout fingerprinting/model loading.
 * Outputs are modified only on success. No network or model parsing occurs. */
sn_status sn_fingerprint_model(const char *model_dir, const char *runtime_precision,
                               uint8_t out32[32]);

/* CSNH v1 header: magic4, u32 version/header_bytes/H/V/heads/bits/reserved,
 * fingerprint32, payload_sha25632. Payload per head: u32 offset/rank,
 * u64 down_bytes/up_bytes, u32 bias_count/reserved, then CSQ1 down/up and
 * little-endian finite F32 bias[V]. No padding/trailing bytes. All heads use
 * header source_bits. File and decoded F32 allocations each cap at 512 MiB.
 * The digest detects byte corruption; it is not an authenticity signature.
 * out must initially be NULL; failed opens leave it unchanged. */
sn_status sn_open(const char *path, const uint8_t expected_fp32[32],
                  size_t hidden_dim, size_t vocab_size, sn_bank **out);
sn_status sn_get_info(const sn_bank *bank, sn_info *out);
sn_status sn_predict(const sn_bank *bank, const float *hidden, int *out_tokens);
/* Explicit capacities and scalar/parallel choice for callers/tests. Hidden and
 * output must be disjoint and genuinely have their declared accessible sizes.
 * Output publication is all-or-none. Concurrent calls are safe: no bank scratch
 * or validation cache is mutated. No allocation or weight rescanning per call. */
sn_status sn_predict_sized(const sn_bank *bank, const float *hidden, size_t hidden_count,
                           int *out_tokens, size_t token_count, int parallel);
void sn_close(sn_bank *bank);
const char *sn_status_string(sn_status status);
#endif

/* Native independent proposals, compiled in a separate non-fast-math object.
 * CSNH v1/v2 store true CSQ1 signed INT4/8/16 weights, decoded ONCE to immutable
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
typedef enum {
    SN_KIND_INVALID = -1, SN_KIND_ORDINARY = 0, SN_KIND_LOGIT_RESIDUAL = 1
} sn_kind;

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

/* CSNH header: magic4, u32 version/header_bytes/H/V/heads/bits/kind,
 * fingerprint32, payload_sha25632. Payload per head: u32 offset/rank,
 * u64 down_bytes/up_bytes, u32 bias_count/reserved, then CSQ1 down/up and
 * little-endian finite F32 bias[V]. No padding/trailing bytes. All heads use
 * header source_bits. File and decoded F32 allocations each cap at 512 MiB.
 * The digest detects byte corruption; it is not an authenticity signature.
 * Accepted (version,kind): (1,0) ordinary; (2,1) cached-logit residual. All
 * other combinations are invalid. out must initially be NULL; failed opens
 * leave it unchanged. */
sn_status sn_open(const char *path, const uint8_t expected_fp32[32],
                  size_t hidden_dim, size_t vocab_size, sn_bank **out);
sn_status sn_get_info(const sn_bank *bank, sn_info *out);
/* Kept separate so sn_info retains its existing C/ctypes ABI. */
sn_kind sn_bank_kind(const sn_bank *bank);
/* Ordinary APIs refuse residual banks. */
sn_status sn_predict(const sn_bank *bank, const float *hidden, int *out_tokens);
/* Explicit capacities and scalar/parallel choice for callers/tests. Hidden and
 * output must be disjoint and genuinely have their declared accessible sizes.
 * Output publication is all-or-none. Concurrent calls are safe: no bank scratch
 * or validation cache is mutated. No allocation or weight rescanning per call. */
sn_status sn_predict_sized(const sn_bank *bank, const float *hidden, size_t hidden_count,
                           int *out_tokens, size_t token_count, int parallel);
/* Residual APIs refuse ordinary banks. Every head reads the SAME immutable
 * features and raw base logits. base_count must equal V exactly; output must
 * be disjoint from both inputs. All inputs must be genuinely accessible for
 * their declared sizes. No output is published on error.
 * Arithmetic: down dot in double -> F32 latent; up dot in double + F32 bias
 * in double + F32 base[v] in double -> checked F32 score, lowest-ID argmax.
 * No intermediate F32 rounding occurs between bias and base additions.
 * Caller guarantees raw base belongs to the same checkpoint/precision,
 * session/frontier h_t BEFORE sampler penalties/masks, whose offset1 is
 * already chosen. Snapshot/rollback must retain matching hidden/base rows.
 * The bank holds no frontier, cache, or mutable prediction scratch. */
sn_status sn_predict_residual(const sn_bank *bank, const float *hidden,
                              const float *base_logits, int *out_tokens);
sn_status sn_predict_residual_sized(const sn_bank *bank, const float *hidden,
                                    size_t hidden_count, const float *base_logits,
                                    size_t base_count, int *out_tokens,
                                    size_t token_count, int parallel);
void sn_close(sn_bank *bank);
const char *sn_status_string(sn_status status);
#endif

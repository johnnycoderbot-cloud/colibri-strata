/* Experimental parallel future-token heads with real INT4/INT8/INT16 STORAGE.
 *
 * CSQ1 down/up blobs hold block-quantized integer weights. This portable CPU
 * reference dequantizes into caller-owned F32 workspace and computes matvecs
 * with double accumulation and F32 activations/output. It is not an all-integer
 * GPU/NPU/TPU kernel and does not establish hardware or model-engine support.
 * Heads independently read the SAME immutable h_t, using the offset convention
 * and bounded dimensions in strata_heads.h. There is no head-to-head loop.
 *
 * Immutable weights/descriptors/hidden are borrowed. Output logits and token
 * ids are TRANSACTIONAL: neither changes unless every head succeeds. Scratch
 * workspace may change on failure. No allocations occur. Greedy ties use the
 * lowest token id. OpenMP parallelizes across heads; otherwise scalar fallback.
 * Every call validates encoded storage/finite values, with real reference cost.
 */
#ifndef COLIBRI_STRATA_QUANT_HEADS_H
#define COLIBRI_STRATA_QUANT_HEADS_H

#include <string.h>
#include "strata_heads.h"
#include "strata_quant.h"

typedef struct {
    size_t future_offset;
    size_t rank;
    const uint8_t *down;        /* CSQ1 row-major [rank, hidden_dim] */
    size_t down_bytes;
    const uint8_t *up;          /* CSQ1 row-major [vocab_size, rank] */
    size_t up_bytes;
    const float *bias;          /* optional F32 [vocab_size] */
    size_t bias_count;
} strata_quant_heads_head;

typedef struct {
    size_t hidden_dim;
    size_t vocab_size;
    size_t head_count;
    const strata_quant_heads_head *heads;
} strata_quant_heads_bank;

static inline coli_strata_status strata_quant_heads_overlap_status_(
        const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    int overlap = strata_heads_overlap(a, a_bytes, b, b_bytes);
    return overlap < 0 ? COLI_STRATA_OVERFLOW : overlap ? COLI_STRATA_INVALID : COLI_STRATA_OK;
}

/* Called only after descriptor dimensions/counts have been checked. */
static inline coli_strata_status strata_quant_heads_output_alias_(
        const strata_quant_heads_bank *bank, const float *hidden, size_t hidden_count,
        const void *output, size_t output_bytes) {
    size_t i;
    coli_strata_status status;
    status = strata_quant_heads_overlap_status_(output, output_bytes, hidden, hidden_count * sizeof(float));
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_overlap_status_(output, output_bytes, bank, sizeof(*bank));
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_overlap_status_(output, output_bytes, bank->heads, bank->head_count * sizeof(*bank->heads));
    if (status != COLI_STRATA_OK) return status;
    for (i = 0; i < bank->head_count; ++i) {
        const strata_quant_heads_head *head = &bank->heads[i];
        status = strata_quant_heads_overlap_status_(output, output_bytes, head->down, head->down_bytes);
        if (status != COLI_STRATA_OK) return status;
        status = strata_quant_heads_overlap_status_(output, output_bytes, head->up, head->up_bytes);
        if (status != COLI_STRATA_OK) return status;
        status = strata_quant_heads_overlap_status_(output, output_bytes, head->bias, head->bias_count * sizeof(float));
        if (status != COLI_STRATA_OK) return status;
    }
    return COLI_STRATA_OK;
}

/* Per-head scratch: max(down_values,up_values) decoded weights, rank latent
 * values, then vocab_size staged scores. This is a conservative CPU reference
 * workspace, not a claim that quantized weights stay compressed during math.
 * Precisions may differ between heads or even between down/up within a head.
 * Result sizes are published together only after full shape/blob validation. */
static inline coli_strata_status strata_quant_heads_sizes(
        const strata_quant_heads_bank *bank, size_t *workspace_floats, size_t *logit_floats) {
    size_t total = 0, logits, i, j;
    coli_strata_status status;
    if (!bank || !bank->heads || !workspace_floats || !logit_floats) return COLI_STRATA_INVALID;
    if (!bank->hidden_dim || bank->hidden_dim > INT_MAX || !bank->vocab_size ||
        bank->vocab_size > INT_MAX || !bank->head_count || bank->head_count > STRATA_HEADS_MAX_HEADS)
        return COLI_STRATA_CAPACITY;
    if (!strata_heads_mul_size(bank->head_count, bank->vocab_size, &logits) ||
        logits > SIZE_MAX / sizeof(float)) return COLI_STRATA_OVERFLOW;
    for (i = 0; i < bank->head_count; ++i) {
        const strata_quant_heads_head *head = &bank->heads[i];
        strata_quant_info down_info, up_info;
        size_t down_values, up_values, scratch;
        if (!head->rank || head->rank > STRATA_HEADS_MAX_RANK || !head->future_offset ||
            head->future_offset > STRATA_HEADS_MAX_OFFSET) return COLI_STRATA_CAPACITY;
        for (j = 0; j < i; ++j)
            if (bank->heads[j].future_offset == head->future_offset) return COLI_STRATA_INVALID;
        if (!strata_heads_mul_size(head->rank, bank->hidden_dim, &down_values) ||
            !strata_heads_mul_size(bank->vocab_size, head->rank, &up_values) ||
            down_values > SIZE_MAX / sizeof(float) || up_values > SIZE_MAX / sizeof(float))
            return COLI_STRATA_OVERFLOW;
        if (!head->down || !head->up || !head->down_bytes || !head->up_bytes ||
            (head->bias ? head->bias_count != bank->vocab_size : head->bias_count != 0))
            return COLI_STRATA_INVALID;
        if (strata_heads_overlap(head->down, head->down_bytes, NULL, 0) < 0 ||
            strata_heads_overlap(head->up, head->up_bytes, NULL, 0) < 0)
            return COLI_STRATA_OVERFLOW;
        status = strata_quant_inspect(head->down, head->down_bytes, &down_info);
        if (status != COLI_STRATA_OK) return status;
        status = strata_quant_inspect(head->up, head->up_bytes, &up_info);
        if (status != COLI_STRATA_OK) return status;
        if (down_info.count != down_values || up_info.count != up_values) return COLI_STRATA_INVALID;
        scratch = down_values > up_values ? down_values : up_values;
        if (head->rank > SIZE_MAX - scratch) return COLI_STRATA_OVERFLOW;
        scratch += head->rank;
        if (bank->vocab_size > SIZE_MAX - scratch) return COLI_STRATA_OVERFLOW;
        scratch += bank->vocab_size;
        if (scratch > SIZE_MAX - total) return COLI_STRATA_OVERFLOW;
        total += scratch;
    }
    if (total > SIZE_MAX / sizeof(float)) return COLI_STRATA_OVERFLOW;
    status = strata_quant_heads_overlap_status_(workspace_floats, sizeof(*workspace_floats),
                                                logit_floats, sizeof(*logit_floats));
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_output_alias_(bank, NULL, 0, workspace_floats, sizeof(*workspace_floats));
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_output_alias_(bank, NULL, 0, logit_floats, sizeof(*logit_floats));
    if (status != COLI_STRATA_OK) return status;
    *workspace_floats = total;
    *logit_floats = logits;
    return COLI_STRATA_OK;
}

static inline coli_strata_status strata_quant_heads_validate(const strata_quant_heads_bank *bank) {
    size_t needed_workspace = 0, needed_logits = 0, i;
    coli_strata_status status = strata_quant_heads_sizes(bank, &needed_workspace, &needed_logits);
    if (status != COLI_STRATA_OK) return status;
    for (i = 0; i < bank->head_count; ++i)
        if (!strata_heads_all_finite(bank->heads[i].bias, bank->heads[i].bias_count)) return COLI_STRATA_NONFINITE;
    return COLI_STRATA_OK;
}

/* logits are [head_index,vocab_token]; tokens are [head_index]. Caller-owned
 * writable buffers must be mutually disjoint and disjoint from all borrowed
 * input. The whole bank is immutable throughout the call, including threads.
 */
static inline coli_strata_status strata_quant_heads_predict(
        const strata_quant_heads_bank *bank, const float *hidden, size_t hidden_count,
        float *workspace, size_t workspace_count, float *logits, size_t logit_count,
        int *tokens, size_t token_count, int parallel) {
    size_t needed_workspace = 0, needed_logits = 0, weight_counts[STRATA_HEADS_MAX_HEADS];
    size_t offsets[STRATA_HEADS_MAX_HEADS], offset = 0;
    int chosen[STRATA_HEADS_MAX_HEADS], i;
    coli_strata_status statuses[STRATA_HEADS_MAX_HEADS], status;
    size_t workspace_bytes, logit_bytes, token_bytes;
    if (!hidden || !workspace || !logits || !tokens) return COLI_STRATA_INVALID;
    status = strata_quant_heads_sizes(bank, &needed_workspace, &needed_logits);
    if (status != COLI_STRATA_OK) return status;
    if (hidden_count != bank->hidden_dim || workspace_count < needed_workspace ||
        logit_count < needed_logits || token_count < bank->head_count) return COLI_STRATA_CAPACITY;
    workspace_bytes = needed_workspace * sizeof(float);
    logit_bytes = needed_logits * sizeof(float);
    token_bytes = bank->head_count * sizeof(int);
    status = strata_quant_heads_output_alias_(bank, hidden, hidden_count, workspace, workspace_bytes);
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_output_alias_(bank, hidden, hidden_count, logits, logit_bytes);
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_output_alias_(bank, hidden, hidden_count, tokens, token_bytes);
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_overlap_status_(workspace, workspace_bytes, logits, logit_bytes);
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_overlap_status_(workspace, workspace_bytes, tokens, token_bytes);
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_overlap_status_(logits, logit_bytes, tokens, token_bytes);
    if (status != COLI_STRATA_OK) return status;
    status = strata_quant_heads_validate(bank);
    if (status != COLI_STRATA_OK) return status;
    if (!strata_heads_all_finite(hidden, hidden_count)) return COLI_STRATA_NONFINITE;
    for (i = 0; i < (int)bank->head_count; ++i) {
        const strata_quant_heads_head *head = &bank->heads[i];
        size_t down_count = head->rank * bank->hidden_dim, up_count = bank->vocab_size * head->rank;
        weight_counts[i] = down_count > up_count ? down_count : up_count;
        offsets[i] = offset;
        offset += weight_counts[i] + head->rank + bank->vocab_size;
        statuses[i] = COLI_STRATA_OK;
    }
#ifndef _OPENMP
    (void)parallel;
#else
#pragma omp parallel for if(parallel)
#endif
    for (i = 0; i < (int)bank->head_count; ++i) {
        const strata_quant_heads_head *head = &bank->heads[i];
        float *weights = workspace + offsets[i];
        float *latent = weights + weight_counts[i];
        float *scores = latent + head->rank;
        float best = 0.0f;
        size_t v;
        int best_token = 0;
        statuses[i] = strata_quant_matvec(head->down, head->down_bytes, head->rank,
                         bank->hidden_dim, hidden, hidden_count, latent, head->rank,
                         weights, weight_counts[i]);
        if (statuses[i] != COLI_STRATA_OK) continue;
        statuses[i] = strata_quant_matvec(head->up, head->up_bytes, bank->vocab_size,
                         head->rank, latent, head->rank, scores, bank->vocab_size,
                         weights, weight_counts[i]);
        if (statuses[i] != COLI_STRATA_OK) continue;
        for (v = 0; v < bank->vocab_size; ++v) {
            double sum = (double)scores[v] + (head->bias ? (double)head->bias[v] : 0.0);
            if (!isfinite(sum)) {
                statuses[i] = COLI_STRATA_NONFINITE;
                break;
            }
            if (sum > FLT_MAX || sum < -FLT_MAX) {
                statuses[i] = COLI_STRATA_OVERFLOW;
                break;
            }
            scores[v] = (float)sum;
            if (!v || scores[v] > best) { best = scores[v]; best_token = (int)v; }
        }
        chosen[i] = best_token;
    }
    for (i = 0; i < (int)bank->head_count; ++i)
        if (statuses[i] != COLI_STRATA_OK) return statuses[i];
    for (i = 0; i < (int)bank->head_count; ++i) {
        const float *scores = workspace + offsets[i] + weight_counts[i] + bank->heads[i].rank;
        memcpy(logits + (size_t)i * bank->vocab_size, scores, bank->vocab_size * sizeof(float));
        tokens[i] = chosen[i];
    }
    return COLI_STRATA_OK;
}

#endif /* COLIBRI_STRATA_QUANT_HEADS_H */

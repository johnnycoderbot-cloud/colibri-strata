/* Experimental architecture-independent future-token heads and greedy verification.
 *
 * Every head reads the SAME immutable hidden vector h_t. A head at offset 1
 * predicts x_(t+1), offset 2 predicts x_(t+2), etc. There is no head-to-head
 * autoregressive loop. Usually the target already supplies offset 1, so trained
 * draft heads start at offset 2. For verification, prepend that target token to
 * the ordered offset-2-and-later predictions to make immediate-next proposals.
 * Offsets must be contiguous for such a proposal; prediction itself permits gaps.
 *
 * Weights are borrowed, immutable, already trained F32 arrays. This header does
 * not train/load weights, discover model hidden states, or integrate any engine.
 * It provides a scalar reference and an OpenMP parallel-for ACROSS heads. It
 * makes no inference-throughput claim. All finite inputs/weights are checked;
 * nonfinite arithmetic is an error, never a silently selected token. Greedy ties
 * choose the lowest token id. Output logits/workspace are invalid after errors.
 *
 * The verifier is a correctness reference: snapshot -> causal batched target ->
 * accept matching prefix -> restore -> REPLAY committed tokens. Restore/replay
 * costs real computation. A production adapter may retain accepted KV/recurrent
 * state only after proving equivalent state/stop semantics for that engine.
 */
#ifndef COLIBRI_STRATA_HEADS_H
#define COLIBRI_STRATA_HEADS_H

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "strata_heads requires finite checks: compile without fast-math/finite-math-only"
#endif

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define STRATA_HEADS_MAX_HEADS 8
#define STRATA_HEADS_MAX_OFFSET (STRATA_HEADS_MAX_HEADS + 1)
#define STRATA_HEADS_MAX_RANK 4096
/* Eight learned offsets 2..9 plus the target's offset1 token. A target
 * adapter may impose a smaller batch limit and must trim the proposal then. */
#define STRATA_HEADS_MAX_DRAFTS (STRATA_HEADS_MAX_HEADS + 1)

typedef enum {
    STRATA_HEADS_OK = 0,
    STRATA_HEADS_EINVAL,
    STRATA_HEADS_EBOUNDS,
    STRATA_HEADS_EOVERFLOW,
    STRATA_HEADS_ENONFINITE,
    STRATA_HEADS_ECALLBACK,
    STRATA_HEADS_EROLLBACK
} strata_heads_status;

typedef struct {
    size_t future_offset;
    size_t rank;
    const float *down;           /* row-major [rank, hidden_dim] */
    size_t down_count;
    const float *up;             /* row-major [vocab_size, rank] */
    size_t up_count;
    const float *bias;           /* optional [vocab_size]; NULL requires count 0 */
    size_t bias_count;
} strata_heads_head;

typedef struct {
    size_t hidden_dim;
    size_t vocab_size;
    size_t head_count;
    const strata_heads_head *heads;
} strata_heads_bank;

static inline int strata_heads_mul_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

/* Shape/count validation without scanning weights. Counts must exactly match
 * dimensions. Caller still guarantees that each declared array is accessible. */
static inline strata_heads_status strata_heads_sizes(
    const strata_heads_bank *bank, size_t *workspace_floats, size_t *logit_floats) {
    size_t latent_count = 0, logits_count, i, j;
    if (!bank || !workspace_floats || !logit_floats || !bank->heads)
        return STRATA_HEADS_EINVAL;
    if (!bank->hidden_dim || bank->hidden_dim > INT_MAX || !bank->vocab_size ||
        bank->vocab_size > INT_MAX || !bank->head_count ||
        bank->head_count > STRATA_HEADS_MAX_HEADS) return STRATA_HEADS_EBOUNDS;
    if (!strata_heads_mul_size(bank->head_count, bank->vocab_size, &logits_count) ||
        logits_count > SIZE_MAX / sizeof(float)) return STRATA_HEADS_EOVERFLOW;
    for (i = 0; i < bank->head_count; ++i) {
        const strata_heads_head *head = &bank->heads[i];
        size_t down_count, up_count;
        if (!head->rank || head->rank > STRATA_HEADS_MAX_RANK ||
            !head->future_offset || head->future_offset > STRATA_HEADS_MAX_OFFSET)
            return STRATA_HEADS_EBOUNDS;
        for (j = 0; j < i; ++j)
            if (bank->heads[j].future_offset == head->future_offset)
                return STRATA_HEADS_EINVAL;
        if (!strata_heads_mul_size(head->rank, bank->hidden_dim, &down_count) ||
            !strata_heads_mul_size(bank->vocab_size, head->rank, &up_count) ||
            down_count > SIZE_MAX / sizeof(float) ||
            up_count > SIZE_MAX / sizeof(float)) return STRATA_HEADS_EOVERFLOW;
        if (!head->down || !head->up || head->down_count != down_count ||
            head->up_count != up_count ||
            (head->bias ? head->bias_count != bank->vocab_size : head->bias_count != 0))
            return STRATA_HEADS_EINVAL;
        if (head->rank > SIZE_MAX - latent_count) return STRATA_HEADS_EOVERFLOW;
        latent_count += head->rank;
    }
    if (latent_count > SIZE_MAX / sizeof(float)) return STRATA_HEADS_EOVERFLOW;
    *workspace_floats = latent_count;
    *logit_floats = logits_count;
    return STRATA_HEADS_OK;
}

static inline int strata_heads_all_finite(const float *values, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) if (!isfinite(values[i])) return 0;
    return 1;
}

static inline strata_heads_status strata_heads_validate(const strata_heads_bank *bank) {
    size_t latent_count, logits_count, i;
    strata_heads_status status = strata_heads_sizes(bank, &latent_count, &logits_count);
    if (status != STRATA_HEADS_OK) return status;
    for (i = 0; i < bank->head_count; ++i) {
        const strata_heads_head *head = &bank->heads[i];
        if (!strata_heads_all_finite(head->down, head->down_count) ||
            !strata_heads_all_finite(head->up, head->up_count) ||
            !strata_heads_all_finite(head->bias, head->bias_count))
            return STRATA_HEADS_ENONFINITE;
    }
    return STRATA_HEADS_OK;
}

/* Return 1 for overlapping byte ranges, -1 for an address/count overflow. This
 * is defensive alias checking, not validation that memory is actually mapped. */
static inline int strata_heads_overlap(const void *a, size_t a_bytes,
                                       const void *b, size_t b_bytes) {
    uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
    if (a_bytes > UINTPTR_MAX - aa || b_bytes > UINTPTR_MAX - bb) return -1;
    if (!a_bytes || !b_bytes) return 0;
    return aa < bb + b_bytes && bb < aa + a_bytes;
}

static inline strata_heads_status strata_heads_check_output_alias(
    const strata_heads_bank *bank, const float *hidden,
    void *output, size_t output_bytes) {
    size_t i;
    int overlap = strata_heads_overlap(output, output_bytes, hidden,
                                      bank->hidden_dim * sizeof(float));
    if (overlap) return overlap < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
    overlap = strata_heads_overlap(output, output_bytes, bank, sizeof(*bank));
    if (overlap) return overlap < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
    overlap = strata_heads_overlap(output, output_bytes, bank->heads,
                                   bank->head_count * sizeof(*bank->heads));
    if (overlap) return overlap < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
    for (i = 0; i < bank->head_count; ++i) {
        const strata_heads_head *head = &bank->heads[i];
        overlap = strata_heads_overlap(output, output_bytes, head->down,
                                       head->down_count * sizeof(float));
        if (overlap) return overlap < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
        overlap = strata_heads_overlap(output, output_bytes, head->up,
                                       head->up_count * sizeof(float));
        if (overlap) return overlap < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
        overlap = strata_heads_overlap(output, output_bytes, head->bias,
                                       head->bias_count * sizeof(float));
        if (overlap) return overlap < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
    }
    return STRATA_HEADS_OK;
}

/* logits[head_index, token_id], tokens[head_index]. Workspace stores one
 * [rank] latent vector per head. All three writable ranges must be disjoint
 * from one another, descriptors, hidden, and weights. No allocations occur.
 * Tokens are published only if every head succeeds. Weights are revalidated on
 * every invocation; this conservative reference has measurable validation cost.
 * Without _OPENMP, parallel is ignored and the same scalar arithmetic is used. */
static inline strata_heads_status strata_heads_predict(
    const strata_heads_bank *bank, const float *hidden, size_t hidden_count,
    float *workspace, size_t workspace_count, float *logits, size_t logits_count,
    int *tokens, size_t token_count, int parallel) {
    size_t needed_workspace, needed_logits, latent_offsets[STRATA_HEADS_MAX_HEADS];
    int chosen[STRATA_HEADS_MAX_HEADS], errors[STRATA_HEADS_MAX_HEADS], i;
    size_t workspace_bytes, logit_bytes, token_bytes, offset = 0;
    strata_heads_status status;
    if (!hidden || !workspace || !logits || !tokens) return STRATA_HEADS_EINVAL;
    status = strata_heads_sizes(bank, &needed_workspace, &needed_logits);
    if (status != STRATA_HEADS_OK) return status;
    if (hidden_count != bank->hidden_dim || workspace_count < needed_workspace ||
        logits_count < needed_logits || token_count < bank->head_count)
        return STRATA_HEADS_EBOUNDS;
    workspace_bytes = needed_workspace * sizeof(float);
    logit_bytes = needed_logits * sizeof(float);
    token_bytes = bank->head_count * sizeof(int);
    status = strata_heads_check_output_alias(bank, hidden, workspace, workspace_bytes);
    if (status != STRATA_HEADS_OK) return status;
    status = strata_heads_check_output_alias(bank, hidden, logits, logit_bytes);
    if (status != STRATA_HEADS_OK) return status;
    status = strata_heads_check_output_alias(bank, hidden, tokens, token_bytes);
    if (status != STRATA_HEADS_OK) return status;
    i = strata_heads_overlap(workspace, workspace_bytes, logits, logit_bytes);
    if (!i) i = strata_heads_overlap(workspace, workspace_bytes, tokens, token_bytes);
    if (!i) i = strata_heads_overlap(logits, logit_bytes, tokens, token_bytes);
    if (i) return i < 0 ? STRATA_HEADS_EOVERFLOW : STRATA_HEADS_EINVAL;
    status = strata_heads_validate(bank);
    if (status != STRATA_HEADS_OK) return status;
    if (!strata_heads_all_finite(hidden, hidden_count)) return STRATA_HEADS_ENONFINITE;
    for (i = 0; i < (int)bank->head_count; ++i) {
        latent_offsets[i] = offset;
        offset += bank->heads[i].rank;
        errors[i] = 0;
    }
#ifndef _OPENMP
    (void)parallel;
#else
#pragma omp parallel for if(parallel)
#endif
    for (i = 0; i < (int)bank->head_count; ++i) {
        const strata_heads_head *head = &bank->heads[i];
        float *latent = workspace + latent_offsets[i];
        float *row = logits + (size_t)i * bank->vocab_size;
        float best = 0.0f;
        size_t r, d, v;
        int best_token = 0;
        for (r = 0; r < head->rank; ++r) {
            float sum = 0.0f;
            for (d = 0; d < bank->hidden_dim; ++d)
                sum += head->down[r * bank->hidden_dim + d] * hidden[d];
            if (!isfinite(sum)) { errors[i] = 1; break; }
            latent[r] = sum;
        }
        if (errors[i]) continue;
        for (v = 0; v < bank->vocab_size; ++v) {
            float sum = head->bias ? head->bias[v] : 0.0f;
            for (r = 0; r < head->rank; ++r)
                sum += head->up[v * head->rank + r] * latent[r];
            if (!isfinite(sum)) { errors[i] = 1; break; }
            row[v] = sum;
            if (!v || sum > best) { best = sum; best_token = (int)v; }
        }
        chosen[i] = best_token;
    }
    for (i = 0; i < (int)bank->head_count; ++i)
        if (errors[i]) return STRATA_HEADS_ENONFINITE;
    for (i = 0; i < (int)bank->head_count; ++i) tokens[i] = chosen[i];
    return STRATA_HEADS_OK;
}

typedef enum {
    STRATA_HEADS_STOP_NONE = 0,
    STRATA_HEADS_STOP_EOS,
    STRATA_HEADS_STOP_TOKEN,
    STRATA_HEADS_STOP_BUDGET,
    STRATA_HEADS_STOP_REQUESTED
} strata_heads_stop_reason;

typedef struct {
    void *ctx;
    /* 0 means success for all fallible callbacks. snapshot failure must leave
     * state unchanged. A successful snapshot stays live until release. */
    int (*snapshot)(void *ctx);
    /* Fill K+1 greedy rows: row i is the next target token after consuming
     * exactly drafts[0..i). Rows must use the same causal state, masks, numeric
     * path and greedy tie policy as plain decoding. Mutations are provisional;
     * caller snapshot/restore must cover KV, recurrent state, position, cache
     * metadata and any other observable model state. K can be zero. */
    int (*batch)(void *ctx, const int *drafts, size_t count,
                 int *target_rows, size_t row_count);
    /* Restore is idempotent, including after a partially failed commit. */
    int (*restore)(void *ctx);
    /* Append/forward ALL tokens including the final correction, bonus or EOS,
     * so future generation/persistence includes every emitted token. */
    int (*commit)(void *ctx, const int *tokens, size_t count);
    void (*release)(void *ctx);
    /* Optional pure predicate for terminal tokens besides eos_token. */
    int (*is_stop_token)(void *ctx, int token);
} strata_heads_greedy_ops;

typedef struct {
    const int *drafts;           /* immediate next tokens, ordered, NOT head offsets */
    size_t draft_count;
    size_t output_budget;       /* remaining tokens allowed, includes EOS/stop */
    size_t vocab_size;
    int eos_token;              /* -1 disables the built-in EOS check */
    int stop_requested;         /* already requested: no callbacks/mutation */
} strata_heads_greedy_request;

typedef struct {
    size_t output_count;
    size_t accepted_count;
    size_t verified_draft_count; /* actual batch depth after budget capping */
    int target_token;           /* correction/bonus, -1 when suppressed by stop */
    strata_heads_stop_reason stop_reason;
} strata_heads_greedy_result;

static inline strata_heads_stop_reason strata_heads_token_stop(
    const strata_heads_greedy_ops *ops, const strata_heads_greedy_request *request,
    int token) {
    if (request->eos_token >= 0 && token == request->eos_token) return STRATA_HEADS_STOP_EOS;
    if (ops->is_stop_token && ops->is_stop_token(ops->ctx, token)) return STRATA_HEADS_STOP_TOKEN;
    return STRATA_HEADS_STOP_NONE;
}

/* Greedy-only: this is NOT stochastic/rejection sampling. Result/output are
 * published only after successful commit; outputs remain untouched on failure.
 * The output may alias drafts since publication follows all callback reads.
 * Capacity must cover min(draft_count+1, output_budget), even if EOS stops early.
 * A failing restore returns EROLLBACK and state is then explicitly untrusted.
 * A batch/commit error with successful rollback leaves original state intact.
 */
static inline strata_heads_status strata_heads_greedy_verify(
    const strata_heads_greedy_ops *ops, const strata_heads_greedy_request *request,
    int *output, size_t output_capacity, strata_heads_greedy_result *result) {
    int rows[STRATA_HEADS_MAX_DRAFTS + 1], committed[STRATA_HEADS_MAX_DRAFTS + 1];
    strata_heads_greedy_result next = {0, 0, 0, -1, STRATA_HEADS_STOP_NONE};
    size_t count, capacity_needed, i;
    strata_heads_status failure;
    if (!ops || !request || !result || !ops->snapshot || !ops->batch ||
        !ops->restore || !ops->commit || !ops->release) return STRATA_HEADS_EINVAL;
    if (request->draft_count > STRATA_HEADS_MAX_DRAFTS || !request->vocab_size ||
        request->vocab_size > INT_MAX || request->eos_token < -1 ||
        (request->eos_token >= 0 && (size_t)request->eos_token >= request->vocab_size))
        return STRATA_HEADS_EBOUNDS;
    if (request->draft_count && !request->drafts) return STRATA_HEADS_EINVAL;
    if (request->stop_requested || !request->output_budget) {
        next.stop_reason = request->stop_requested ? STRATA_HEADS_STOP_REQUESTED : STRATA_HEADS_STOP_BUDGET;
        *result = next;
        return STRATA_HEADS_OK;
    }
    for (i = 0; i < request->draft_count; ++i)
        if (request->drafts[i] < 0 || (size_t)request->drafts[i] >= request->vocab_size)
            return STRATA_HEADS_EBOUNDS;
    count = request->draft_count;
    if (count >= request->output_budget) count = request->output_budget - 1;
    capacity_needed = count + 1;
    if (!output || output_capacity < capacity_needed) return STRATA_HEADS_EBOUNDS;
    next.verified_draft_count = count;
    if (ops->snapshot(ops->ctx)) return STRATA_HEADS_ECALLBACK;
    if (ops->batch(ops->ctx, request->drafts, count, rows, count + 1)) {
        failure = STRATA_HEADS_ECALLBACK;
        goto strata_heads_rollback;
    }
    for (i = 0; i <= count; ++i)
        if (rows[i] < 0 || (size_t)rows[i] >= request->vocab_size) {
            failure = STRATA_HEADS_EBOUNDS;
            goto strata_heads_rollback;
        }
    for (i = 0; i < count && request->drafts[i] == rows[i]; ++i) {
        committed[next.output_count++] = request->drafts[i];
        ++next.accepted_count;
        next.stop_reason = strata_heads_token_stop(ops, request, request->drafts[i]);
        if (next.stop_reason != STRATA_HEADS_STOP_NONE) break;
    }
    if (next.stop_reason == STRATA_HEADS_STOP_NONE) {
        next.target_token = rows[next.accepted_count];
        committed[next.output_count++] = next.target_token;
        next.stop_reason = strata_heads_token_stop(ops, request, next.target_token);
    }
    if (next.stop_reason == STRATA_HEADS_STOP_NONE && next.output_count == request->output_budget)
        next.stop_reason = STRATA_HEADS_STOP_BUDGET;
    if (ops->restore(ops->ctx)) {
        ops->release(ops->ctx);
        return STRATA_HEADS_EROLLBACK;
    }
    if (ops->commit(ops->ctx, committed, next.output_count)) {
        failure = STRATA_HEADS_ECALLBACK;
        goto strata_heads_rollback;
    }
    ops->release(ops->ctx);
    for (i = 0; i < next.output_count; ++i) output[i] = committed[i];
    *result = next;
    return STRATA_HEADS_OK;
strata_heads_rollback:
    if (ops->restore(ops->ctx)) failure = STRATA_HEADS_EROLLBACK;
    ops->release(ops->ctx);
    return failure;
}

#endif /* COLIBRI_STRATA_HEADS_H */

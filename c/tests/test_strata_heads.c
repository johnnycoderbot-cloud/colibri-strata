#include <float.h>
#include <stdio.h>
#include <string.h>
#include "../strata_heads.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

static int test_predictions(void) {
    float down[3][2] = {{1, 0}, {0, 1}, {-1, 1}};
    float up[3][4] = {{0, 2, 1, -1}, {0, -1, 3, 1}, {0, 1, 2, 4}};
    float bias[4] = {0, 0, 0, 0};
    strata_heads_head heads[3] = {
        {2, 1, down[0], 2, up[0], 4, NULL, 0},
        {3, 1, down[1], 2, up[1], 4, bias, 4},
        {5, 1, down[2], 2, up[2], 4, NULL, 0}
    };
    strata_heads_bank bank = {2, 4, 3, heads};
    float hidden[2] = {2, 3}, saved_hidden[2], work_a[3], work_b[3];
    float logits_a[12], logits_b[12];
    const float expected[12] = {0, 4, 2, -2, 0, -3, 9, 3, 0, 1, 2, 4};
    int tokens_a[3], tokens_b[3], before[3];
    size_t work_count = 0, logit_count = 0;
    strata_heads_status status;
    memcpy(saved_hidden, hidden, sizeof(hidden));
    CHECK(strata_heads_sizes(&bank, &work_count, &logit_count) == STRATA_HEADS_OK);
    CHECK(work_count == 3 && logit_count == 12);
    CHECK(strata_heads_validate(&bank) == STRATA_HEADS_OK);
    CHECK(strata_heads_predict(&bank, hidden, 2, work_a, 3, logits_a, 12, tokens_a, 3, 0) == STRATA_HEADS_OK);
    CHECK(strata_heads_predict(&bank, hidden, 2, work_b, 3, logits_b, 12, tokens_b, 3, 1) == STRATA_HEADS_OK);
    CHECK(memcmp(logits_a, expected, sizeof(expected)) == 0);
    CHECK(memcmp(logits_a, logits_b, sizeof(logits_a)) == 0);
    CHECK(memcmp(work_a, work_b, sizeof(work_a)) == 0);
    CHECK(memcmp(tokens_a, tokens_b, sizeof(tokens_a)) == 0);
    CHECK(tokens_a[0] == 1 && tokens_a[1] == 2 && tokens_a[2] == 3);
    CHECK(memcmp(hidden, saved_hidden, sizeof(hidden)) == 0);
    /* Changing the first head cannot feed its predicted token into later heads. */
    up[0][3] = 10;
    CHECK(strata_heads_predict(&bank, hidden, 2, work_b, 3, logits_b, 12, tokens_b, 3, 1) == STRATA_HEADS_OK);
    CHECK(tokens_b[0] == 3 && tokens_b[1] == 2 && tokens_b[2] == 3);
    CHECK(memcmp(logits_b + 4, expected + 4, 8 * sizeof(float)) == 0);
    up[0][3] = -1;
    /* All zero logits exercise the documented lowest-id greedy tie. */
    hidden[0] = hidden[1] = 0;
    CHECK(strata_heads_predict(&bank, hidden, 2, work_b, 3, logits_b, 12, tokens_b, 3, 1) == STRATA_HEADS_OK);
    CHECK(tokens_b[0] == 0 && tokens_b[1] == 0 && tokens_b[2] == 0);
    hidden[0] = 2; hidden[1] = 3;
    memcpy(before, tokens_a, sizeof(before));
#define EXPECT_PREDICT_ERROR(expected_status) do { \
    status = strata_heads_predict(&bank, hidden, 2, work_a, 3, logits_a, 12, tokens_a, 3, 1); \
    CHECK(status == (expected_status)); CHECK(memcmp(tokens_a, before, sizeof(before)) == 0); \
} while (0)
    heads[0].down_count = 1; EXPECT_PREDICT_ERROR(STRATA_HEADS_EINVAL); heads[0].down_count = 2;
    heads[0].up = NULL; EXPECT_PREDICT_ERROR(STRATA_HEADS_EINVAL); heads[0].up = up[0];
    heads[1].bias_count = 3; EXPECT_PREDICT_ERROR(STRATA_HEADS_EINVAL); heads[1].bias_count = 4;
    heads[0].bias_count = 1; EXPECT_PREDICT_ERROR(STRATA_HEADS_EINVAL); heads[0].bias_count = 0;
    heads[0].rank = 0; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); heads[0].rank = 1;
    heads[0].rank = STRATA_HEADS_MAX_RANK + 1; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); heads[0].rank = 1;
    heads[0].future_offset = 0; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); heads[0].future_offset = 2;
    heads[0].future_offset = STRATA_HEADS_MAX_OFFSET + 1; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); heads[0].future_offset = 2;
    heads[1].future_offset = 2; EXPECT_PREDICT_ERROR(STRATA_HEADS_EINVAL); heads[1].future_offset = 3;
    bank.head_count = 0; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); bank.head_count = 3;
    bank.head_count = STRATA_HEADS_MAX_HEADS + 1; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); bank.head_count = 3;
    bank.vocab_size = (size_t)INT_MAX + 1; EXPECT_PREDICT_ERROR(STRATA_HEADS_EBOUNDS); bank.vocab_size = 4;
    hidden[0] = NAN; EXPECT_PREDICT_ERROR(STRATA_HEADS_ENONFINITE); hidden[0] = 2;
    down[0][0] = INFINITY; EXPECT_PREDICT_ERROR(STRATA_HEADS_ENONFINITE); down[0][0] = 1;
    up[1][1] = NAN; EXPECT_PREDICT_ERROR(STRATA_HEADS_ENONFINITE); up[1][1] = -1;
    bias[0] = NAN; EXPECT_PREDICT_ERROR(STRATA_HEADS_ENONFINITE); bias[0] = 0;
    down[0][0] = FLT_MAX; EXPECT_PREDICT_ERROR(STRATA_HEADS_ENONFINITE); down[0][0] = 1;
    up[0][1] = FLT_MAX; EXPECT_PREDICT_ERROR(STRATA_HEADS_ENONFINITE); up[0][1] = 2;
    CHECK(strata_heads_predict(&bank, hidden, 1, work_a, 3, logits_a, 12, tokens_a, 3, 0) == STRATA_HEADS_EBOUNDS);
    CHECK(strata_heads_predict(&bank, hidden, 2, work_a, 2, logits_a, 12, tokens_a, 3, 0) == STRATA_HEADS_EBOUNDS);
    CHECK(strata_heads_predict(&bank, hidden, 2, work_a, 3, logits_a, 11, tokens_a, 3, 0) == STRATA_HEADS_EBOUNDS);
    CHECK(strata_heads_predict(&bank, hidden, 2, work_a, 3, logits_a, 12, tokens_a, 2, 0) == STRATA_HEADS_EBOUNDS);
    CHECK(strata_heads_predict(&bank, hidden, 2, hidden, 3, logits_a, 12, tokens_a, 3, 0) == STRATA_HEADS_EINVAL);
    CHECK(strata_heads_predict(&bank, hidden, 2, logits_a, 3, logits_a, 12, tokens_a, 3, 0) == STRATA_HEADS_EINVAL);
    CHECK(strata_heads_predict(&bank, hidden, 2, down[0], 3, logits_a, 12, tokens_a, 3, 0) == STRATA_HEADS_EINVAL);
    CHECK(!strata_heads_mul_size(SIZE_MAX, 2, &work_count));
    CHECK(strata_heads_overlap((const void *)(uintptr_t)(UINTPTR_MAX - 2), 4, hidden, sizeof(hidden)) == -1);
#undef EXPECT_PREDICT_ERROR
    return 0;
}

static int test_mixed_ranks(void) {
    const float hidden[3] = {1, 2, 3};
    const float down_a[6] = {1, 0, 0, 0, 1, 1};
    const float down_b[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float up_a[8] = {1, 0, 0, 1, 1, 1, -1, 2};
    const float up_b[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    const float bias[4] = {0, 10, 0, 0};
    const float expected[8] = {1, 5, 6, 9, 1, 12, 3, 6};
    const float expected_latent[5] = {1, 5, 1, 2, 3};
    strata_heads_head heads[STRATA_HEADS_MAX_HEADS] = {
        {1, 2, down_a, 6, up_a, 8, NULL, 0},
        {9, 3, down_b, 9, up_b, 12, bias, 4}
    };
    strata_heads_bank bank = {3, 4, 2, heads};
    float work[STRATA_HEADS_MAX_HEADS * 3], logits[STRATA_HEADS_MAX_HEADS * 4];
    int tokens[STRATA_HEADS_MAX_HEADS];
    size_t i;
    CHECK(strata_heads_predict(&bank, hidden, 3, work, 24, logits, 32, tokens, 8, 1) == STRATA_HEADS_OK);
    CHECK(memcmp(logits, expected, sizeof(expected)) == 0);
    CHECK(memcmp(work, expected_latent, sizeof(expected_latent)) == 0);
    CHECK(tokens[0] == 3 && tokens[1] == 1);
    /* Exercise every allowed parallel head, with variable workspace offsets. */
    for (i = 2; i < STRATA_HEADS_MAX_HEADS; ++i) heads[i] = heads[i % 2];
    for (i = 0; i < STRATA_HEADS_MAX_HEADS; ++i) heads[i].future_offset = i + 1;
    bank.head_count = STRATA_HEADS_MAX_HEADS;
    CHECK(strata_heads_predict(&bank, hidden, 3, work, 24, logits, 32, tokens, 8, 1) == STRATA_HEADS_OK);
    for (i = 0; i < STRATA_HEADS_MAX_HEADS; ++i) {
        CHECK(tokens[i] == (i % 2 ? 1 : 3));
        CHECK(memcmp(logits + i * 4, expected + (i % 2) * 4, 4 * sizeof(float)) == 0);
    }
    return 0;
}

/* A causal target with both token history and recurrent state. Changing any
 * prior token changes its next-token computation; rollback must restore BOTH.
 * Test batch intentionally evaluates every proposed row, including rejected
 * suffixes, so preserving a provisional state cannot accidentally pass. */
enum { TOY_VOCAB = 9, TOY_EOS = 8, TOY_STOP = 7, TOY_HISTORY = 128 };
typedef struct {
    int history[TOY_HISTORY];
    size_t length;
    uint32_t hash, recurrent;
} toy_state;

typedef struct {
    toy_state state, saved;
    int snapshot_live, snapshots, restores, batches, commits, releases;
    int fail_snapshot, fail_batch, fail_restore, fail_commit, bad_row;
    size_t eos_at, stop_at;
} toy_target;

static void toy_append(toy_target *toy, int token) {
    if (toy->state.length >= TOY_HISTORY) { fputs("toy history overflow\n", stderr); return; }
    toy->state.history[toy->state.length++] = token;
    toy->state.hash = toy->state.hash * UINT32_C(1664525) + (uint32_t)token + UINT32_C(1013904223);
    toy->state.recurrent = (toy->state.recurrent << 3) ^ (toy->state.hash >> 2) ^ (uint32_t)token;
}

static int toy_next(const toy_target *toy) {
    if (toy->state.length == toy->eos_at) return TOY_EOS;
    if (toy->state.length == toy->stop_at) return TOY_STOP;
    return (int)((toy->state.hash % 7 + toy->state.recurrent % 7 + toy->state.length % 7) % 7);
}

static void toy_init(toy_target *toy) {
    memset(toy, 0, sizeof(*toy));
    toy->eos_at = toy->stop_at = SIZE_MAX;
    toy_append(toy, 3);
    toy_append(toy, 1);
}

static int toy_snapshot(void *ctx) {
    toy_target *toy = ctx;
    ++toy->snapshots;
    if (toy->fail_snapshot || toy->snapshot_live) return 1;
    toy->saved = toy->state;
    toy->snapshot_live = 1;
    return 0;
}

static int toy_batch(void *ctx, const int *drafts, size_t count, int *rows, size_t row_count) {
    toy_target *toy = ctx;
    size_t i;
    ++toy->batches;
    if (!toy->snapshot_live || row_count != count + 1) return 1;
    for (i = 0; i <= count; ++i) {
        rows[i] = toy_next(toy);
        if (i < count) toy_append(toy, drafts[i]);
        if (toy->fail_batch && i == 0) return 1;
    }
    if (toy->bad_row) rows[count] = TOY_VOCAB;
    return 0;
}

static int toy_restore(void *ctx) {
    toy_target *toy = ctx;
    ++toy->restores;
    if (toy->fail_restore || !toy->snapshot_live) return 1;
    toy->state = toy->saved;
    return 0;
}

static int toy_commit(void *ctx, const int *tokens, size_t count) {
    toy_target *toy = ctx;
    size_t i;
    ++toy->commits;
    for (i = 0; i < count; ++i) {
        toy_append(toy, tokens[i]);
        if (toy->fail_commit && i == 0) return 1;
    }
    return 0;
}

static void toy_release(void *ctx) {
    toy_target *toy = ctx;
    ++toy->releases;
    toy->snapshot_live = 0;
}

static int toy_is_stop(void *ctx, int token) { (void)ctx; return token == TOY_STOP; }

static strata_heads_greedy_ops toy_ops(toy_target *toy) {
    strata_heads_greedy_ops ops = {
        toy, toy_snapshot, toy_batch, toy_restore, toy_commit, toy_release, toy_is_stop
    };
    return ops;
}

static size_t toy_plain(toy_target *toy, int *output, size_t budget) {
    size_t count = 0;
    while (count < budget) {
        int token = toy_next(toy);
        output[count++] = token;
        toy_append(toy, token);
        if (token == TOY_EOS || token == TOY_STOP) break;
    }
    return count;
}

static void toy_make_drafts(const toy_target *toy, int *drafts, size_t count) {
    toy_target copy = *toy;
    size_t i;
    for (i = 0; i < count; ++i) {
        drafts[i] = toy_next(&copy);
        toy_append(&copy, drafts[i]);
    }
}

static int verify_case(int mismatch, size_t budget, size_t terminal_at, int terminal,
                       size_t expected_accepted, size_t expected_count,
                       strata_heads_stop_reason expected_reason) {
    toy_target toy, plain;
    int drafts[4], output[5] = {-1, -1, -1, -1, -1}, expected[5];
    strata_heads_greedy_request request;
    strata_heads_greedy_result result;
    strata_heads_greedy_ops ops;
    toy_init(&toy);
    if (terminal == TOY_EOS) toy.eos_at = terminal_at;
    if (terminal == TOY_STOP) toy.stop_at = terminal_at;
    plain = toy;
    toy_make_drafts(&toy, drafts, 4);
    if (mismatch >= 0) drafts[mismatch] = (drafts[mismatch] + 1) % 7;
    request.drafts = drafts;
    request.draft_count = 4;
    request.output_budget = budget;
    request.vocab_size = TOY_VOCAB;
    request.eos_token = TOY_EOS;
    request.stop_requested = 0;
    ops = toy_ops(&toy);
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 5, &result) == STRATA_HEADS_OK);
    CHECK(result.accepted_count == expected_accepted && result.output_count == expected_count);
    CHECK(result.stop_reason == expected_reason);
    CHECK(toy_plain(&plain, expected, result.output_count) == result.output_count);
    CHECK(memcmp(output, expected, result.output_count * sizeof(int)) == 0);
    CHECK(memcmp(&toy.state, &plain.state, sizeof(toy.state)) == 0);
    CHECK(!toy.snapshot_live && toy.snapshots == 1 && toy.restores == 1 && toy.releases == 1);
    CHECK(toy.batches == 1 && toy.commits == 1);
    if (expected_reason == STRATA_HEADS_STOP_EOS || expected_reason == STRATA_HEADS_STOP_TOKEN) {
        CHECK(output[result.output_count - 1] == terminal);
        if (result.accepted_count == result.output_count) CHECK(result.target_token == -1);
    } else CHECK(result.target_token == output[result.output_count - 1]);
    return 0;
}

static int test_transactions(void) {
    toy_target toy;
    toy_state before_state;
    strata_heads_greedy_ops ops;
    strata_heads_greedy_request request;
    strata_heads_greedy_result result, before_result;
    int drafts[4], output[5], before_output[5], mode;
    /* First/middle rejection and full prefix acceptance must match plain state. */
    CHECK(verify_case(0, 20, SIZE_MAX, -1, 0, 1, STRATA_HEADS_STOP_NONE) == 0);
    CHECK(verify_case(2, 20, SIZE_MAX, -1, 2, 3, STRATA_HEADS_STOP_NONE) == 0);
    CHECK(verify_case(-1, 20, SIZE_MAX, -1, 4, 5, STRATA_HEADS_STOP_NONE) == 0);
    CHECK(verify_case(-1, 1, SIZE_MAX, -1, 0, 1, STRATA_HEADS_STOP_BUDGET) == 0);
    CHECK(verify_case(-1, 3, SIZE_MAX, -1, 2, 3, STRATA_HEADS_STOP_BUDGET) == 0);
    /* EOS as accepted draft, correction, and bonus; no suffix becomes live. */
    CHECK(verify_case(-1, 20, 3, TOY_EOS, 2, 2, STRATA_HEADS_STOP_EOS) == 0);
    CHECK(verify_case(1, 20, 3, TOY_EOS, 1, 2, STRATA_HEADS_STOP_EOS) == 0);
    CHECK(verify_case(-1, 20, 6, TOY_EOS, 4, 5, STRATA_HEADS_STOP_EOS) == 0);
    CHECK(verify_case(-1, 2, 3, TOY_EOS, 1, 2, STRATA_HEADS_STOP_EOS) == 0);
    CHECK(verify_case(-1, 20, 3, TOY_STOP, 2, 2, STRATA_HEADS_STOP_TOKEN) == 0);
    CHECK(verify_case(1, 20, 3, TOY_STOP, 1, 2, STRATA_HEADS_STOP_TOKEN) == 0);
    for (mode = 0; mode < 5; ++mode) {
        strata_heads_status expected_status = mode == 3 ? STRATA_HEADS_EBOUNDS :
                                            mode == 4 ? STRATA_HEADS_EROLLBACK : STRATA_HEADS_ECALLBACK;
        toy_init(&toy);
        toy_make_drafts(&toy, drafts, 4);
        before_state = toy.state;
        memset(&result, 0x5a, sizeof(result)); before_result = result;
        memset(output, 0x5a, sizeof(output)); memcpy(before_output, output, sizeof(output));
        request.drafts = drafts; request.draft_count = 4; request.output_budget = 20;
        request.vocab_size = TOY_VOCAB; request.eos_token = TOY_EOS; request.stop_requested = 0;
        if (mode == 0) toy.fail_snapshot = 1;
        if (mode == 1) toy.fail_batch = 1;
        if (mode == 2) toy.fail_commit = 1;
        if (mode == 3) toy.bad_row = 1;
        if (mode == 4) toy.fail_restore = 1;
        ops = toy_ops(&toy);
        CHECK(strata_heads_greedy_verify(&ops, &request, output, 5, &result) == expected_status);
        CHECK(memcmp(output, before_output, sizeof(output)) == 0);
        CHECK(memcmp(&result, &before_result, sizeof(result)) == 0);
        if (mode != 4) CHECK(memcmp(&toy.state, &before_state, sizeof(before_state)) == 0);
        CHECK(!toy.snapshot_live);
        CHECK(toy.releases == (mode == 0 ? 0 : 1));
        if (mode == 2) CHECK(toy.restores == 2 && toy.commits == 1);
    }
    toy_init(&toy); ops = toy_ops(&toy); toy_make_drafts(&toy, drafts, 4);
    before_state = toy.state;
    request.drafts = drafts; request.draft_count = 4; request.output_budget = 0;
    request.vocab_size = TOY_VOCAB; request.eos_token = TOY_EOS; request.stop_requested = 0;
    CHECK(strata_heads_greedy_verify(&ops, &request, NULL, 0, &result) == STRATA_HEADS_OK);
    CHECK(result.output_count == 0 && result.stop_reason == STRATA_HEADS_STOP_BUDGET);
    request.output_budget = 20; request.stop_requested = 1;
    CHECK(strata_heads_greedy_verify(&ops, &request, NULL, 0, &result) == STRATA_HEADS_OK);
    CHECK(result.output_count == 0 && result.stop_reason == STRATA_HEADS_STOP_REQUESTED);
    CHECK(toy.snapshots == 0 && memcmp(&toy.state, &before_state, sizeof(before_state)) == 0);
    request.stop_requested = 0;
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 4, &result) == STRATA_HEADS_EBOUNDS);
    drafts[0] = -1;
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 5, &result) == STRATA_HEADS_EBOUNDS);
    drafts[0] = TOY_VOCAB;
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 5, &result) == STRATA_HEADS_EBOUNDS);
    request.draft_count = STRATA_HEADS_MAX_DRAFTS + 1;
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 5, &result) == STRATA_HEADS_EBOUNDS);
    request.draft_count = 0; request.drafts = NULL; request.eos_token = TOY_VOCAB;
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 5, &result) == STRATA_HEADS_EBOUNDS);
    request.eos_token = TOY_EOS;
    CHECK(strata_heads_greedy_verify(&ops, &request, output, 1, &result) == STRATA_HEADS_OK);
    CHECK(result.accepted_count == 0 && result.output_count == 1);
    return 0;
}

/* Multi-transaction continuation compares a full 25-token stream and all state.
 * Deterministic wrong proposals force repeated rollback/correction transitions. */
static int test_continuation(void) {
    toy_target toy, plain;
    strata_heads_greedy_ops ops;
    strata_heads_greedy_request request;
    strata_heads_greedy_result result;
    int actual[25], expected[25], drafts[4];
    size_t count = 0, step = 0;
    toy_init(&toy); plain = toy; ops = toy_ops(&toy);
    CHECK(toy_plain(&plain, expected, 25) == 25);
    while (count < 25) {
        toy_make_drafts(&toy, drafts, 4);
        if (step % 3 != 2) {
            size_t bad = step % 3 == 0 ? 0 : 2;
            drafts[bad] = (drafts[bad] + 1) % 7;
        }
        request.drafts = drafts; request.draft_count = 4;
        request.output_budget = 25 - count; request.vocab_size = TOY_VOCAB;
        request.eos_token = TOY_EOS; request.stop_requested = 0;
        CHECK(strata_heads_greedy_verify(&ops, &request, actual + count, 25 - count, &result) == STRATA_HEADS_OK);
        CHECK(result.output_count > 0);
        count += result.output_count;
        ++step;
    }
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    CHECK(memcmp(&toy.state, &plain.state, sizeof(toy.state)) == 0);
    CHECK(result.stop_reason == STRATA_HEADS_STOP_BUDGET && toy.restores > 3);
    return 0;
}

static int test_max_depth_and_alias(void) {
    toy_target toy, plain;
    strata_heads_greedy_ops ops;
    strata_heads_greedy_request request;
    strata_heads_greedy_result result;
    int drafts_and_output[STRATA_HEADS_MAX_DRAFTS + 1], expected[STRATA_HEADS_MAX_DRAFTS + 1];
    toy_init(&toy); plain = toy; ops = toy_ops(&toy);
    toy_make_drafts(&toy, drafts_and_output, STRATA_HEADS_MAX_DRAFTS);
    request.drafts = drafts_and_output; request.draft_count = STRATA_HEADS_MAX_DRAFTS;
    request.output_budget = STRATA_HEADS_MAX_DRAFTS + 1; request.vocab_size = TOY_VOCAB;
    request.eos_token = TOY_EOS; request.stop_requested = 0;
    CHECK(strata_heads_greedy_verify(&ops, &request, drafts_and_output,
                                    STRATA_HEADS_MAX_DRAFTS + 1, &result) == STRATA_HEADS_OK);
    CHECK(result.accepted_count == STRATA_HEADS_MAX_DRAFTS);
    CHECK(result.output_count == STRATA_HEADS_MAX_DRAFTS + 1);
    CHECK(result.stop_reason == STRATA_HEADS_STOP_BUDGET);
    CHECK(toy_plain(&plain, expected, STRATA_HEADS_MAX_DRAFTS + 1) == STRATA_HEADS_MAX_DRAFTS + 1);
    CHECK(memcmp(drafts_and_output, expected, sizeof(expected)) == 0);
    CHECK(memcmp(&toy.state, &plain.state, sizeof(toy.state)) == 0);
    return 0;
}

int main(void) {
    CHECK(test_predictions() == 0);
    CHECK(test_mixed_ranks() == 0);
    CHECK(test_transactions() == 0);
    CHECK(test_continuation() == 0);
    CHECK(test_max_depth_and_alias() == 0);
#ifdef _OPENMP
    puts("strata_heads: prediction and causal transaction tests passed (OpenMP)");
#else
    puts("strata_heads: prediction and causal transaction tests passed (scalar)");
#endif
    return 0;
}

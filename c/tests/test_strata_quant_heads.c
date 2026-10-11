#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../strata_quant_heads.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

enum { HIDDEN = 3, VOCAB = 5, MAX_LOGITS = STRATA_HEADS_MAX_HEADS * VOCAB, SCRATCH = 1024 };

typedef struct {
    strata_quant_heads_head heads[STRATA_HEADS_MAX_HEADS];
    strata_heads_head reference_heads[STRATA_HEADS_MAX_HEADS];
    uint8_t *down[STRATA_HEADS_MAX_HEADS], *up[STRATA_HEADS_MAX_HEADS];
    float *decoded_down[STRATA_HEADS_MAX_HEADS], *decoded_up[STRATA_HEADS_MAX_HEADS];
    float bias[STRATA_HEADS_MAX_HEADS][VOCAB], hidden[HIDDEN];
    strata_quant_heads_bank bank;
    strata_heads_bank reference;
} fixture;

static void fixture_free(fixture *f) {
    size_t h;
    for (h = 0; h < STRATA_HEADS_MAX_HEADS; ++h) {
        free(f->down[h]); free(f->up[h]);
        free(f->decoded_down[h]); free(f->decoded_up[h]);
    }
}

/* Uniform bits=4/8/16, or bits=0 for mixed down/up precisions. Quantized
 * comparisons use the exact same decoded values in the existing F32 heads,
 * rather than comparing lossily quantized weights to their original floats. */
static int fixture_init(fixture *f, size_t head_count, unsigned bits) {
    static const unsigned precisions[3] = {4, 8, 16};
    static const size_t ranks[3] = {3, 1, 2};
    static const float down[3][9] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1},
        {0.25f, -0.5f, 1},
        {1, 0.1f, -0.2f, -0.5f, 2, 1}
    };
    static const float up[3][15] = {
        {1, 0, 0, 0, 2, 0, 0, 0, 4, 1, 1, 1, -1, -1, -1},
        {0, 1, -2, 3, 1},
        {1, 1, 2, -1, -1, 1, 0, 2, 1, 0}
    };
    size_t h;
    memset(f, 0, sizeof(*f));
    f->hidden[0] = 1; f->hidden[1] = 2; f->hidden[2] = 3;
    f->bank.hidden_dim = HIDDEN; f->bank.vocab_size = VOCAB;
    f->bank.head_count = head_count; f->bank.heads = f->heads;
    f->reference.hidden_dim = HIDDEN; f->reference.vocab_size = VOCAB;
    f->reference.head_count = head_count; f->reference.heads = f->reference_heads;
    for (h = 0; h < head_count; ++h) {
        size_t kind = h % 3, rank = ranks[kind], down_count = rank * HIDDEN, up_count = rank * VOCAB;
        size_t down_bytes, up_bytes, written, decoded;
        unsigned down_bits = bits ? bits : precisions[h % 3];
        unsigned up_bits = bits ? bits : precisions[(h + 1) % 3];
        CHECK(strata_quant_size(down_count, 5, down_bits, &down_bytes) == COLI_STRATA_OK);
        CHECK(strata_quant_size(up_count, 5, up_bits, &up_bytes) == COLI_STRATA_OK);
        f->down[h] = malloc(down_bytes); f->up[h] = malloc(up_bytes);
        f->decoded_down[h] = malloc(down_count * sizeof(float));
        f->decoded_up[h] = malloc(up_count * sizeof(float));
        CHECK(f->down[h] && f->up[h] && f->decoded_down[h] && f->decoded_up[h]);
        CHECK(strata_quant_encode(down[kind], down_count, 5, down_bits, f->down[h], down_bytes, &written) == COLI_STRATA_OK);
        CHECK(written == down_bytes);
        CHECK(strata_quant_encode(up[kind], up_count, 5, up_bits, f->up[h], up_bytes, &written) == COLI_STRATA_OK);
        CHECK(written == up_bytes);
        CHECK(strata_quant_decode(f->down[h], down_bytes, f->decoded_down[h], down_count, &decoded) == COLI_STRATA_OK);
        CHECK(decoded == down_count);
        CHECK(strata_quant_decode(f->up[h], up_bytes, f->decoded_up[h], up_count, &decoded) == COLI_STRATA_OK);
        CHECK(decoded == up_count);
        f->bias[h][3] = 0.25f;
        f->heads[h].future_offset = h + 2; f->heads[h].rank = rank;
        f->heads[h].down = f->down[h]; f->heads[h].down_bytes = down_bytes;
        f->heads[h].up = f->up[h]; f->heads[h].up_bytes = up_bytes;
        f->heads[h].bias = f->bias[h]; f->heads[h].bias_count = VOCAB;
        f->reference_heads[h].future_offset = h + 2; f->reference_heads[h].rank = rank;
        f->reference_heads[h].down = f->decoded_down[h]; f->reference_heads[h].down_count = down_count;
        f->reference_heads[h].up = f->decoded_up[h]; f->reference_heads[h].up_count = up_count;
        f->reference_heads[h].bias = f->bias[h]; f->reference_heads[h].bias_count = VOCAB;
    }
    return 0;
}

static int compare_reference(fixture *f) {
    float work[SCRATCH], parallel_work[SCRATCH], reference_work[STRATA_HEADS_MAX_HEADS * 3];
    float logits[MAX_LOGITS], parallel_logits[MAX_LOGITS], reference_logits[MAX_LOGITS];
    float saved_hidden[HIDDEN];
    int tokens[STRATA_HEADS_MAX_HEADS], parallel_tokens[STRATA_HEADS_MAX_HEADS], reference_tokens[STRATA_HEADS_MAX_HEADS];
    size_t needed_workspace, needed_logits, h, v;
    CHECK(strata_quant_heads_sizes(&f->bank, &needed_workspace, &needed_logits) == COLI_STRATA_OK);
    CHECK(needed_workspace <= SCRATCH && needed_logits == f->bank.head_count * VOCAB);
    CHECK(strata_quant_heads_validate(&f->bank) == COLI_STRATA_OK);
    memcpy(saved_hidden, f->hidden, sizeof(saved_hidden));
    CHECK(strata_quant_heads_predict(&f->bank, f->hidden, HIDDEN, work, SCRATCH,
                                     logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_OK);
    CHECK(strata_quant_heads_predict(&f->bank, f->hidden, HIDDEN, parallel_work, SCRATCH,
                                     parallel_logits, MAX_LOGITS, parallel_tokens, STRATA_HEADS_MAX_HEADS, 1) == COLI_STRATA_OK);
    CHECK(strata_heads_predict(&f->reference, f->hidden, HIDDEN, reference_work, STRATA_HEADS_MAX_HEADS * 3,
                               reference_logits, MAX_LOGITS, reference_tokens, STRATA_HEADS_MAX_HEADS, 0) == STRATA_HEADS_OK);
    CHECK(memcmp(logits, parallel_logits, needed_logits * sizeof(float)) == 0);
    CHECK(memcmp(tokens, parallel_tokens, f->bank.head_count * sizeof(int)) == 0);
    CHECK(memcmp(f->hidden, saved_hidden, sizeof(saved_hidden)) == 0);
    for (h = 0; h < f->bank.head_count; ++h) {
        CHECK(tokens[h] == reference_tokens[h]);
        CHECK(tokens[h] == (h % 3 == 0 ? 2 : 3));
        for (v = 0; v < VOCAB; ++v) {
            size_t index = h * VOCAB + v;
            /* Existing F32 heads accumulate in float; this quantized reference
             * accumulates in double, so compare logits with an explicit bound. */
            float tolerance = 2e-5f * (1.0f + fabsf(reference_logits[index]));
            CHECK(fabsf(logits[index] - reference_logits[index]) <= tolerance);
        }
    }
    return 0;
}

static int test_precisions(void) {
    fixture f;
    unsigned bits;
    for (bits = 4; bits <= 16; bits *= 2) {
        CHECK(fixture_init(&f, 3, bits) == 0);
        CHECK(compare_reference(&f) == 0);
        fixture_free(&f);
    }
    CHECK(fixture_init(&f, STRATA_HEADS_MAX_HEADS, 0) == 0);
    CHECK(compare_reference(&f) == 0);
    /* All-zero activations+bias test exact greedy ties across every precision. */
    memset(f.hidden, 0, sizeof(f.hidden));
    memset(f.bias, 0, sizeof(f.bias));
    {
        float work[SCRATCH], logits[MAX_LOGITS];
        int tokens[STRATA_HEADS_MAX_HEADS];
        size_t h;
        CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                                         logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 1) == COLI_STRATA_OK);
        for (h = 0; h < STRATA_HEADS_MAX_HEADS; ++h) CHECK(tokens[h] == 0);
    }
    fixture_free(&f);
    return 0;
}

static int test_rejections(void) {
    fixture f;
    float work[SCRATCH], logits[MAX_LOGITS], saved_logits[MAX_LOGITS];
    int tokens[STRATA_HEADS_MAX_HEADS], saved_tokens[STRATA_HEADS_MAX_HEADS];
    size_t i, needed_workspace, needed_logits, saved_rank, saved_offset;
    uint8_t saved_byte;
    coli_strata_status status;
    CHECK(fixture_init(&f, STRATA_HEADS_MAX_HEADS, 0) == 0);
    for (i = 0; i < MAX_LOGITS; ++i) logits[i] = 42.25f;
    for (i = 0; i < STRATA_HEADS_MAX_HEADS; ++i) tokens[i] = -123;
    memcpy(saved_logits, logits, sizeof(logits)); memcpy(saved_tokens, tokens, sizeof(tokens));
    CHECK(strata_quant_heads_sizes(&f.bank, &needed_workspace, &needed_logits) == COLI_STRATA_OK);
#define EXPECT_ERROR(expected) do { \
    status = strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH, \
                  logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 1); \
    CHECK(status == (expected)); CHECK(memcmp(logits, saved_logits, sizeof(logits)) == 0); \
    CHECK(memcmp(tokens, saved_tokens, sizeof(tokens)) == 0); \
} while (0)
    saved_rank = f.heads[0].rank; f.heads[0].rank = 2;
    EXPECT_ERROR(COLI_STRATA_INVALID); f.heads[0].rank = saved_rank;
    f.heads[0].rank = 0; EXPECT_ERROR(COLI_STRATA_CAPACITY); f.heads[0].rank = saved_rank;
    f.heads[0].rank = STRATA_HEADS_MAX_RANK + 1;
    EXPECT_ERROR(COLI_STRATA_CAPACITY); f.heads[0].rank = saved_rank;
    f.bank.hidden_dim = 4; EXPECT_ERROR(COLI_STRATA_INVALID); f.bank.hidden_dim = HIDDEN;
    f.bank.vocab_size = 4; EXPECT_ERROR(COLI_STRATA_INVALID); f.bank.vocab_size = VOCAB;
    f.bank.head_count = STRATA_HEADS_MAX_HEADS + 1;
    EXPECT_ERROR(COLI_STRATA_CAPACITY); f.bank.head_count = STRATA_HEADS_MAX_HEADS;
    saved_offset = f.heads[1].future_offset; f.heads[1].future_offset = f.heads[0].future_offset;
    EXPECT_ERROR(COLI_STRATA_INVALID); f.heads[1].future_offset = saved_offset;
    f.heads[0].future_offset = STRATA_HEADS_MAX_OFFSET + 1;
    EXPECT_ERROR(COLI_STRATA_CAPACITY); f.heads[0].future_offset = 2;
    f.heads[0].bias_count = 4; EXPECT_ERROR(COLI_STRATA_INVALID); f.heads[0].bias_count = VOCAB;
    f.bias[7][4] = NAN; EXPECT_ERROR(COLI_STRATA_NONFINITE); f.bias[7][4] = 0;
    f.hidden[1] = INFINITY; EXPECT_ERROR(COLI_STRATA_NONFINITE); f.hidden[1] = 2;
    /* Corrupt only the final head: prior output rows still cannot publish. */
    saved_byte = f.up[7][0]; f.up[7][0] = 'X';
    EXPECT_ERROR(COLI_STRATA_INVALID); f.up[7][0] = saved_byte;
    --f.heads[7].up_bytes; EXPECT_ERROR(COLI_STRATA_INVALID); ++f.heads[7].up_bytes;
    /* Head0 down is INT4 with an odd first block of 5 values. Padding must be
     * zero even though the unneeded high nibble could be ignored by a decoder. */
    saved_byte = f.down[0][STRATA_QUANT_HEADER_BYTES + 4 + 2];
    f.down[0][STRATA_QUANT_HEADER_BYTES + 4 + 2] |= 0x10;
    EXPECT_ERROR(COLI_STRATA_INVALID); f.down[0][STRATA_QUANT_HEADER_BYTES + 4 + 2] = saved_byte;
    /* Correct structure but nonfinite serialized maxabs must also be rejected. */
    {
        uint8_t scale[4];
        memcpy(scale, f.down[0] + STRATA_QUANT_HEADER_BYTES, 4);
        coli_strata_put_u32_(f.down[0] + STRATA_QUANT_HEADER_BYTES, UINT32_C(0x7fc00000));
        EXPECT_ERROR(COLI_STRATA_INVALID);
        memcpy(f.down[0] + STRATA_QUANT_HEADER_BYTES, scale, 4);
    }
    /* All descriptors/encodings are valid; final head fails during arithmetic.
     * Other heads can have completed in parallel, but final outputs stay intact. */
    {
        float large[HIDDEN] = {FLT_MAX, 0, 0};
        uint8_t *large_blob;
        size_t bytes, written, original_bytes = f.heads[7].down_bytes;
        CHECK(strata_quant_size(HIDDEN, 5, 8, &bytes) == COLI_STRATA_OK);
        large_blob = malloc(bytes); CHECK(large_blob);
        CHECK(strata_quant_encode(large, HIDDEN, 5, 8, large_blob, bytes, &written) == COLI_STRATA_OK);
        f.heads[7].down = large_blob; f.heads[7].down_bytes = bytes; f.hidden[0] = 2;
        EXPECT_ERROR(COLI_STRATA_OVERFLOW);
        /* Both matvecs remain finite, then adding a finite F32 bias overflows
         * the final logits in the last head. Publication must still be atomic. */
        large[0] = FLT_MAX / 4;
        CHECK(strata_quant_encode(large, HIDDEN, 5, 8, large_blob, bytes, &written) == COLI_STRATA_OK);
        f.hidden[0] = 1; f.bias[7][3] = FLT_MAX;
        EXPECT_ERROR(COLI_STRATA_OVERFLOW);
        f.bias[7][3] = 0.25f;
        f.heads[7].down = f.down[7]; f.heads[7].down_bytes = original_bytes; f.hidden[0] = 1;
        free(large_blob);
    }
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN - 1, work, SCRATCH,
                   logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_CAPACITY);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, needed_workspace - 1,
                   logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_CAPACITY);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                   logits, needed_logits - 1, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_CAPACITY);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                   logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS - 1, 0) == COLI_STRATA_CAPACITY);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                   work, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_INVALID);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                   f.hidden, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_INVALID);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                   logits, MAX_LOGITS, (int *)logits, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_INVALID);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, work, SCRATCH,
                   (float *)f.down[0], MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_INVALID);
    CHECK(strata_quant_heads_predict(&f.bank, f.hidden, HIDDEN, (float *)f.down[0], SCRATCH,
                   logits, MAX_LOGITS, tokens, STRATA_HEADS_MAX_HEADS, 0) == COLI_STRATA_INVALID);
    CHECK(memcmp(logits, saved_logits, sizeof(logits)) == 0 && memcmp(tokens, saved_tokens, sizeof(tokens)) == 0);
    CHECK(strata_quant_heads_sizes(&f.bank, &needed_workspace, &needed_workspace) == COLI_STRATA_INVALID);
    CHECK(strata_quant_heads_sizes(&f.bank, &f.bank.hidden_dim, &needed_logits) == COLI_STRATA_INVALID);
    CHECK(f.bank.hidden_dim == HIDDEN);
    CHECK(compare_reference(&f) == 0); /* Recovered valid bank after all failures. */
#undef EXPECT_ERROR
    fixture_free(&f);
    return 0;
}

int main(void) {
    CHECK(test_precisions() == 0);
    CHECK(test_rejections() == 0);
#ifdef _OPENMP
    puts("strata_quant_heads: INT4/INT8/INT16 and transactional tests passed (OpenMP)");
#else
    puts("strata_quant_heads: INT4/INT8/INT16 and transactional tests passed (scalar)");
#endif
    return 0;
}

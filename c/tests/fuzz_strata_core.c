/* Deterministic adversarial properties and a libFuzzer entry point.
 * Standalone: cc -std=c99 -O1 -g -fsanitize=address,undefined \
 *   fuzz_strata_core.c -lm -o fuzz_strata_core; ./fuzz_strata_core 100000 1
 * libFuzzer: clang -DSTRATA_LIBFUZZER -fsanitize=fuzzer,address,undefined \
 *   -O1 -g fuzz_strata_core.c -lm -o fuzz_strata_core
 * Every pointer/capacity describes real accessible storage. No tests manufacture
 * invalid addresses. Random inputs exercise errors as well as successful paths;
 * passing proves these cases, not absence of every possible bug or GPU parity.
 */
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../strata_heads.h"
#include "../strata_quant.h"
#include "../strata_quant_heads.h"

enum { FUZZ_VALUES = 129, FUZZ_BYTES = 1024, FUZZ_STEPS = 24 };
static uint64_t fuzz_case_seed;
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "fuzz_strata_core: seed=%" PRIu64 " %s:%d: %s\n", \
            fuzz_case_seed, __FILE__, __LINE__, #c); abort(); \
} } while (0)

typedef struct { uint64_t before; float values[FUZZ_VALUES]; uint64_t after; } float_guard;
typedef struct { uint64_t before; uint8_t values[FUZZ_BYTES]; uint64_t after; } byte_guard;
static const uint64_t guard_word = UINT64_C(0xf3b567e9c21da480);

static uint64_t random64(uint64_t *r) {
    uint64_t x = *r ? *r : UINT64_C(0x6a09e667f3bcc909);
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *r = x;
    return x * UINT64_C(2685821657736338717);
}
static void float_guard_init(float_guard *g) {
    memset(g, 0, sizeof(*g)); g->before = g->after = guard_word;
    for (size_t i = 0; i < FUZZ_VALUES; ++i) g->values[i] = 913.25f;
}
static void byte_guard_init(byte_guard *g) {
    memset(g, 0x5a, sizeof(*g)); g->before = g->after = guard_word;
}
static void guards_ok(const float_guard *f, const byte_guard *b) {
    if (f) REQUIRE(f->before == guard_word && f->after == guard_word);
    if (b) REQUIRE(b->before == guard_word && b->after == guard_word);
}

static void random_values(float *v, size_t count, uint64_t *r) {
    for (size_t i = 0; i < count; ++i) {
        switch (random64(r) % 12) {
        case 0: v[i] = 0; break;
        case 1: v[i] = -0.0f; break;
        case 2: v[i] = nextafterf(0, 1); break;
        case 3: v[i] = -nextafterf(0, 1); break;
        case 4: v[i] = FLT_MAX; break;
        case 5: v[i] = -FLT_MAX; break;
        default: v[i] = (float)((int)(random64(r) % 20001) - 10000) / 97.0f; break;
        }
    }
}

/* Deliberately damage lengths, metadata, scales and reserved codes, and
 * retain some undamaged inputs so each fuzz run reaches valid decode paths. */
static size_t mutate(byte_guard *b, size_t length, int quant, uint64_t *r) {
    unsigned mode = (unsigned)(random64(r) % 8);
    if (!mode) return length;
    if (mode == 1 && length) b->values[random64(r) % length] ^= (uint8_t)(1u << (random64(r) % 8));
    else if (mode == 2) return length ? (size_t)(random64(r) % length) : 0;
    else if (mode == 3 && length + 7 < FUZZ_BYTES) {
        size_t n = 1 + (size_t)(random64(r) % 7);
        for (size_t i = 0; i < n; ++i) b->values[length + i] = (uint8_t)random64(r);
        return length + n;
    } else if (mode == 4) {
        size_t field = (quant ? 6 : 4), at = (size_t)(random64(r) % field) * 4;
        coli_strata_put_u32_(b->values + at, (uint32_t)random64(r));
    } else if (mode == 5 && length > (size_t)(quant ? 28 : 20)) {
        size_t at = quant ? 28 : 20;
        b->values[at] = (uint8_t)(quant ? 0x88 : 128);
    } else if (mode == 6 && length >= (size_t)(quant ? 28 : 20)) {
        const uint32_t scales[] = {0x7fc00000u, 0x7f800000u, 0xbf800000u, 0u};
        coli_strata_put_u32_(b->values + (quant ? 24 : 16), scales[random64(r) % 4]);
    } else if (mode == 7) {
        length = (size_t)(random64(r) % FUZZ_BYTES);
        for (size_t i = 0; i < length; ++i) b->values[i] = (uint8_t)random64(r);
    }
    return length;
}

static void check_decode(const uint8_t *data, size_t length, size_t capacity, int quant) {
    float_guard output, before;
    size_t count = 777;
    coli_strata_status status;
    float_guard_init(&output); before = output;
    status = quant ? strata_quant_decode(data, length, output.values, capacity, &count)
                   : coli_strata_kv_decode(data, length, output.values, capacity, &count);
    guards_ok(&output, NULL);
    if (status != COLI_STRATA_OK) {
        REQUIRE(count == 777 && !memcmp(&output, &before, sizeof(output)));
    } else {
        REQUIRE(count <= capacity && count <= FUZZ_VALUES);
        for (size_t i = 0; i < count; ++i) REQUIRE(isfinite(output.values[i]));
        for (size_t i = count; i < FUZZ_VALUES; ++i) REQUIRE(output.values[i] == before.values[i]);
    }
    if (quant) {
        strata_quant_info info, saved;
        memset(&info, 0x5a, sizeof(info)); saved = info;
        status = strata_quant_inspect(data, length, &info);
        if (status != COLI_STRATA_OK) REQUIRE(!memcmp(&info, &saved, sizeof(info)));
        else {
            REQUIRE(info.bits == 4 || info.bits == 8 || info.bits == 16);
            REQUIRE(info.encoded_bytes == length && info.count <= STRATA_QUANT_MAX_VALUES);
            REQUIRE(info.block_values > 0 && info.block_values <= STRATA_QUANT_MAX_BLOCK);
            REQUIRE(info.block_count == info.count / info.block_values + (info.count % info.block_values != 0));
        }
    }
}

static void codec_case(uint64_t *r, int quant) {
    float_guard input, restored;
    byte_guard encoded, before, mutated, readonly;
    size_t count = (size_t)(random64(r) % (FUZZ_VALUES + 1));
    size_t block = 1 + (size_t)(random64(r) % 65), written = 999, needed = 999;
    unsigned precisions[] = {4, 8, 16}, bits = precisions[random64(r) % 3];
    unsigned qmax = bits == 4 ? 7 : bits == 8 ? 127 : 32767;
    coli_strata_status status;
    float_guard_init(&input); float_guard_init(&restored);
    byte_guard_init(&encoded); random_values(input.values, count, r);
    REQUIRE((quant ? strata_quant_size(count, block, bits, &needed)
                   : coli_strata_kv_size(count, block, &needed)) == COLI_STRATA_OK);
    REQUIRE(needed <= FUZZ_BYTES);
    REQUIRE((quant ? strata_quant_encode(input.values, count, block, bits, encoded.values, FUZZ_BYTES, &written)
                   : coli_strata_kv_encode(input.values, count, block, encoded.values, FUZZ_BYTES, &written)) == COLI_STRATA_OK);
    REQUIRE(written == needed); guards_ok(&input, &encoded); readonly = encoded;
    {
        size_t decoded = 999;
        REQUIRE((quant ? strata_quant_decode(encoded.values, written, restored.values, FUZZ_VALUES, &decoded)
                       : coli_strata_kv_decode(encoded.values, written, restored.values, FUZZ_VALUES, &decoded)) == COLI_STRATA_OK);
        REQUIRE(decoded == count); guards_ok(&restored, NULL);
        for (size_t start = 0; start < count; start += block) {
            size_t end = start + block < count ? start + block : count;
            double maximum = 0;
            for (size_t i = start; i < end; ++i) if (fabs((double)input.values[i]) > maximum) maximum = fabs((double)input.values[i]);
            for (size_t i = start; i < end; ++i) {
                double bound = maximum / (2 * (quant ? qmax : 127)) + maximum * FLT_EPSILON + 2 * (double)nextafterf(0, 1);
                REQUIRE(isfinite(restored.values[i]));
                REQUIRE(fabs((double)input.values[i] - restored.values[i]) <= bound);
            }
        }
    }
    REQUIRE(!memcmp(&encoded, &readonly, sizeof(encoded)));
    mutated = encoded;
    {
        size_t length = mutate(&mutated, written, quant, r);
        byte_guard original = mutated;
        check_decode(mutated.values, length, (size_t)(random64(r) % (FUZZ_VALUES + 1)), quant);
        REQUIRE(!memcmp(&mutated, &original, sizeof(mutated))); guards_ok(NULL, &mutated);
    }
    /* Every original encoding is also checked at every structural boundary. */
    check_decode(encoded.values, written ? written - 1 : 0, FUZZ_VALUES, quant);
    check_decode(encoded.values, 0, FUZZ_VALUES, quant);
    before = encoded; written = 999;
    if (count) input.values[random64(r) % count] = (random64(r) & 1) ? NAN : INFINITY;
    status = quant ? strata_quant_encode(input.values, count, block, bits, encoded.values, needed - 1, &written)
                   : coli_strata_kv_encode(input.values, count, block, encoded.values, needed - 1, &written);
    REQUIRE(status != COLI_STRATA_OK && written == 999 && !memcmp(&encoded, &before, sizeof(encoded)));
    if (count) {
        status = quant ? strata_quant_encode(input.values, count, block, bits, encoded.values, FUZZ_BYTES, &written)
                       : coli_strata_kv_encode(input.values, count, block, encoded.values, FUZZ_BYTES, &written);
        REQUIRE(status == COLI_STRATA_NONFINITE && written == 999 && !memcmp(&encoded, &before, sizeof(encoded)));
    }
}

static void matvec_case(uint64_t *r) {
    float_guard weights, decoded, x, y, before_y, workspace;
    byte_guard blob, saved_blob;
    size_t rows = 1 + (size_t)(random64(r) % 8), cols = 1 + (size_t)(random64(r) % 16);
    size_t count = rows * cols, written = 0, decoded_count = 0;
    unsigned precision[] = {4, 8, 16}, bits = precision[random64(r) % 3];
    float_guard_init(&weights); float_guard_init(&decoded); float_guard_init(&x);
    float_guard_init(&y); float_guard_init(&workspace); byte_guard_init(&blob);
    random_values(weights.values, count, r);
    for (size_t i = 0; i < cols; ++i) x.values[i] = (float)((int)(random64(r) % 2001) - 1000) / 113.0f;
    REQUIRE(strata_quant_encode(weights.values, count, 1 + (size_t)(random64(r) % 65),
            bits, blob.values, FUZZ_BYTES, &written) == COLI_STRATA_OK);
    REQUIRE(strata_quant_decode(blob.values, written, decoded.values, FUZZ_VALUES, &decoded_count) == COLI_STRATA_OK);
    REQUIRE(decoded_count == count); saved_blob = blob; before_y = y;
    {
        coli_strata_status s = strata_quant_matvec(blob.values, written, rows, cols,
                x.values, cols, y.values, FUZZ_VALUES, workspace.values, FUZZ_VALUES);
        int overflow = 0;
        float expected[8];
        for (size_t row = 0; row < rows; ++row) {
            double sum = 0;
            for (size_t col = 0; col < cols; ++col)
                sum += (double)decoded.values[row * cols + col] * x.values[col];
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) overflow = 1;
            expected[row] = overflow ? 0 : (float)sum;
        }
        if (overflow) {
            REQUIRE(s == COLI_STRATA_OVERFLOW && !memcmp(&y, &before_y, sizeof(y)));
        } else {
            REQUIRE(s == COLI_STRATA_OK);
            for (size_t row = 0; row < rows; ++row) REQUIRE(y.values[row] == expected[row]);
            for (size_t row = rows; row < FUZZ_VALUES; ++row) REQUIRE(y.values[row] == before_y.values[row]);
        }
    }
    REQUIRE(!memcmp(&blob, &saved_blob, sizeof(blob))); guards_ok(&y, &blob); guards_ok(&workspace, NULL);
    before_y = y; x.values[random64(r) % cols] = NAN;
    REQUIRE(strata_quant_matvec(blob.values, written, rows, cols, x.values, cols,
            y.values, FUZZ_VALUES, workspace.values, FUZZ_VALUES) == COLI_STRATA_NONFINITE);
    REQUIRE(!memcmp(&y, &before_y, sizeof(y)));
    x.values[0] = INFINITY;
    REQUIRE(strata_quant_matvec(blob.values, written, rows, cols, x.values, cols,
            y.values, FUZZ_VALUES, workspace.values, FUZZ_VALUES) == COLI_STRATA_NONFINITE);
    REQUIRE(!memcmp(&y, &before_y, sizeof(y)));
    for (size_t col = 0; col < cols; ++col) x.values[col] = 0;
    REQUIRE(strata_quant_matvec(blob.values, written, rows, cols, x.values, cols,
            y.values, rows - 1, workspace.values, FUZZ_VALUES) != COLI_STRATA_OK);
    REQUIRE(!memcmp(&y, &before_y, sizeof(y)));
    REQUIRE(strata_quant_matvec(blob.values, written, rows, cols, x.values, cols,
            y.values, FUZZ_VALUES, workspace.values, count - 1) != COLI_STRATA_OK);
    REQUIRE(!memcmp(&y, &before_y, sizeof(y)));
    {
        size_t length = mutate(&blob, written, 1, r);
        coli_strata_status s = strata_quant_matvec(blob.values, length, rows, cols,
                x.values, cols, y.values, FUZZ_VALUES, workspace.values, FUZZ_VALUES);
        if (s != COLI_STRATA_OK) REQUIRE(!memcmp(&y, &before_y, sizeof(y)));
        else for (size_t row = 0; row < rows; ++row) REQUIRE(y.values[row] == 0);
        guards_ok(&y, &blob); guards_ok(&workspace, NULL);
    }
}

static void quant_heads_case(uint64_t *r) {
    enum { H = 4, V = 7, BANK_WORK = 512 };
    uint8_t down[8][256], up[8][256];
    float biases[8][V], hidden[H], workspace[BANK_WORK], logits[8 * V], saved_logits[8 * V];
    int tokens[8], saved_tokens[8];
    strata_quant_heads_head hs[8];
    size_t head_count = 1 + (size_t)(random64(r) % 8), w = 0, l = 0;
    strata_quant_heads_bank bank = {H, V, head_count, hs};
    unsigned precisions[] = {4, 8, 16};
    memset(hs, 0, sizeof(hs));
    for (size_t i = 0; i < H; ++i) hidden[i] = (float)((int)(random64(r) % 2001) - 1000) / 317.0f;
    for (size_t k = 0; k < head_count; ++k) {
        float raw[28];
        size_t rank = 1 + (size_t)(random64(r) % 4), bytes = 0;
        hs[k].rank = rank; hs[k].future_offset = k + 2;
        for (size_t i = 0; i < rank * H; ++i) raw[i] = (float)((int)(random64(r) % 2001) - 1000) / 419.0f;
        REQUIRE(strata_quant_encode(raw, rank * H, 1 + (size_t)(random64(r) % 9),
                precisions[random64(r) % 3], down[k], sizeof(down[k]), &bytes) == COLI_STRATA_OK);
        hs[k].down = down[k]; hs[k].down_bytes = bytes;
        for (size_t i = 0; i < rank * V; ++i) raw[i] = (float)((int)(random64(r) % 2001) - 1000) / 421.0f;
        REQUIRE(strata_quant_encode(raw, rank * V, 1 + (size_t)(random64(r) % 9),
                precisions[random64(r) % 3], up[k], sizeof(up[k]), &bytes) == COLI_STRATA_OK);
        hs[k].up = up[k]; hs[k].up_bytes = bytes;
        for (size_t i = 0; i < V; ++i) biases[k][i] = (float)((int)(random64(r) % 101) - 50) / 113.0f;
        hs[k].bias = biases[k]; hs[k].bias_count = V;
    }
    REQUIRE(strata_quant_heads_sizes(&bank, &w, &l) == COLI_STRATA_OK && w <= BANK_WORK && l == head_count * V);
    REQUIRE(strata_quant_heads_predict(&bank, hidden, H, workspace, BANK_WORK,
            logits, 8 * V, tokens, 8, 0) == COLI_STRATA_OK);
    for (size_t k = 0; k < head_count; ++k) {
        float wd[16], wu[28], latent[4]; size_t decoded = 0; int best = 0;
        REQUIRE(strata_quant_decode(down[k], hs[k].down_bytes, wd, 16, &decoded) == COLI_STRATA_OK);
        REQUIRE(decoded == hs[k].rank * H);
        REQUIRE(strata_quant_decode(up[k], hs[k].up_bytes, wu, 28, &decoded) == COLI_STRATA_OK);
        REQUIRE(decoded == hs[k].rank * V);
        for (size_t i = 0; i < hs[k].rank; ++i) {
            double sum = 0; for (size_t d = 0; d < H; ++d) sum += (double)wd[i * H + d] * hidden[d];
            latent[i] = (float)sum;
        }
        for (size_t v = 0; v < V; ++v) {
            double sum = 0; float expected;
            for (size_t i = 0; i < hs[k].rank; ++i) sum += (double)wu[v * hs[k].rank + i] * latent[i];
            expected = (float)((double)(float)sum + biases[k][v]);
            REQUIRE(logits[k * V + v] == expected);
            if (v && expected > logits[k * V + (size_t)best]) best = (int)v;
        }
        REQUIRE(tokens[k] == best);
    }
    memcpy(saved_logits, logits, head_count * V * sizeof(float));
    memcpy(saved_tokens, tokens, head_count * sizeof(int));
    biases[head_count - 1][V - 1] = NAN;
    REQUIRE(strata_quant_heads_predict(&bank, hidden, H, workspace, BANK_WORK,
            logits, 8 * V, tokens, 8, 0) == COLI_STRATA_NONFINITE);
    REQUIRE(!memcmp(saved_logits, logits, head_count * V * sizeof(float)) &&
            !memcmp(saved_tokens, tokens, head_count * sizeof(int)));
    biases[head_count - 1][V - 1] = 0;
    up[head_count - 1][0] = 'X';
    REQUIRE(strata_quant_heads_predict(&bank, hidden, H, workspace, BANK_WORK,
            logits, 8 * V, tokens, 8, 0) == COLI_STRATA_INVALID);
    REQUIRE(!memcmp(saved_logits, logits, head_count * V * sizeof(float)) &&
            !memcmp(saved_tokens, tokens, head_count * sizeof(int)));
}

typedef struct { int transfer[96], consumer[96]; unsigned enqueued, consumed; int reject; } fake_backend;
static int fake_enqueue(void *ctx, unsigned slot, const coli_strata_prefetch_identity *id, uint64_t *event) {
    fake_backend *f = ctx; (void)slot; (void)id;
    if (f->reject) return 0;
    REQUIRE(f->enqueued < 96); *event = f->enqueued++; return 1;
}
static int fake_transfer(void *ctx, uint64_t event) {
    fake_backend *f = ctx; REQUIRE(event < f->enqueued); return f->transfer[event];
}
static int fake_consume(void *ctx, unsigned slot, const coli_strata_prefetch_identity *id, uint64_t *event) {
    fake_backend *f = ctx; (void)slot; (void)id;
    if (f->reject) return 0;
    REQUIRE(f->consumed < 96); *event = f->consumed++; return 1;
}
static int fake_consumer(void *ctx, uint64_t event) {
    fake_backend *f = ctx; REQUIRE(event < f->consumed); return f->consumer[event];
}

static void prefetch_case(uint64_t *r) {
    fake_backend fake = {{0}, {0}, 0, 0, 0};
    coli_strata_prefetch pool;
    coli_strata_prefetch_backend ops = {&fake, fake_enqueue, fake_transfer, fake_consume, fake_consumer};
    coli_strata_prefetch_ticket tickets[2] = {{2,0}, {2,0}}, stale = {2,0};
    REQUIRE(coli_strata_prefetch_init(&pool, 2, 3, 64, 64, &ops) == COLI_STRATA_OK);
    for (unsigned step = 0; step < FUZZ_STEPS; ++step) {
        unsigned choice = (unsigned)(random64(r) % 5), slot = (unsigned)(random64(r) % 2);
        coli_strata_prefetch_slot before = pool.slots[slot];
        if (choice == 0) {
            coli_strata_prefetch_identity id = {step % 2, step % 3, step + 1, 1 + (size_t)(random64(r) % 80)};
            coli_strata_prefetch_ticket out = {2,0};
            fake.reject = (int)(random64(r) % 4 == 0);
            coli_strata_status s = coli_strata_prefetch_enqueue(&pool, &id, &out);
            if (s == COLI_STRATA_OK) { tickets[out.slot] = out; REQUIRE(pool.slots[out.slot].state == COLI_STRATA_SLOT_IN_FLIGHT); }
            else REQUIRE(out.slot == 2 && out.generation == 0);
        } else if (choice == 1) {
            for (unsigned i = 0; i < 2; ++i) if (pool.slots[i].state == COLI_STRATA_SLOT_IN_FLIGHT) {
                int possibilities[] = {0, 1, -1, 2, -2};
                fake.transfer[pool.slots[i].transfer_event] = possibilities[random64(r) % 5];
            }
            (void)coli_strata_prefetch_poll(&pool);
            for (unsigned i = 0; i < 2; ++i) if (pool.slots[i].state == COLI_STRATA_SLOT_READY)
                REQUIRE(fake.transfer[pool.slots[i].transfer_event] == 1);
        } else if (choice == 2 && before.state != COLI_STRATA_SLOT_FREE) {
            coli_strata_prefetch_identity id = before.identity;
            coli_strata_prefetch_ticket out = {2,0};
            if (random64(r) & 1) ++id.tag;
            fake.reject = (int)(random64(r) % 4 == 0);
            coli_strata_status s = coli_strata_prefetch_acquire(&pool, &id, &out);
            if (s == COLI_STRATA_OK) {
                REQUIRE(before.state == COLI_STRATA_SLOT_READY && !before.discard);
                REQUIRE(pool.slots[out.slot].state == COLI_STRATA_SLOT_IN_USE);
                tickets[out.slot] = out;
            } else REQUIRE(out.slot == 2 && out.generation == 0);
        } else if (choice == 3) {
            coli_strata_status s = coli_strata_prefetch_cancel(&pool, tickets[slot]);
            if (before.state == COLI_STRATA_SLOT_IN_FLIGHT || before.state == COLI_STRATA_SLOT_IN_USE) {
                REQUIRE(s == COLI_STRATA_PENDING && pool.slots[slot].state == before.state && pool.slots[slot].discard);
            }
            if (s == COLI_STRATA_CANCELLED) stale = tickets[slot];
        } else if (choice == 4 && before.state == COLI_STRATA_SLOT_IN_USE) {
            int possibilities[] = {0, 1, -1, 2, -2};
            int terminal = possibilities[random64(r) % 5];
            fake.consumer[before.consumer_event] = terminal;
            coli_strata_status s = coli_strata_prefetch_release(&pool, tickets[slot]);
            if (terminal != 1 && terminal != -1) REQUIRE(pool.slots[slot].state == before.state);
            else { REQUIRE(pool.slots[slot].state == COLI_STRATA_SLOT_FREE); stale = tickets[slot]; }
            if (!terminal) REQUIRE(s == COLI_STRATA_PENDING);
        }
        if (stale.slot < 2 && pool.slots[stale.slot].generation != stale.generation) {
            coli_strata_prefetch saved = pool;
            REQUIRE(coli_strata_prefetch_cancel(&pool, stale) == COLI_STRATA_INVALID);
            REQUIRE(!memcmp(&pool, &saved, sizeof(pool)));
        }
    }
    /* Drain both fences. Outstanding work is never recycled by cancellation. */
    for (unsigned i = 0; i < 2; ++i) {
        if (pool.slots[i].state == COLI_STRATA_SLOT_IN_FLIGHT) {
            REQUIRE(coli_strata_prefetch_cancel(&pool, tickets[i]) == COLI_STRATA_PENDING);
            fake.transfer[pool.slots[i].transfer_event] = 1;
        }
        if (pool.slots[i].state == COLI_STRATA_SLOT_IN_USE) {
            REQUIRE(coli_strata_prefetch_cancel(&pool, tickets[i]) == COLI_STRATA_PENDING);
            fake.consumer[pool.slots[i].consumer_event] = 1;
            REQUIRE(coli_strata_prefetch_release(&pool, tickets[i]) == COLI_STRATA_CANCELLED);
        }
        if (pool.slots[i].state == COLI_STRATA_SLOT_READY)
            REQUIRE(coli_strata_prefetch_cancel(&pool, tickets[i]) == COLI_STRATA_CANCELLED);
    }
    (void)coli_strata_prefetch_poll(&pool);
    REQUIRE(pool.slots[0].state == COLI_STRATA_SLOT_FREE && pool.slots[1].state == COLI_STRATA_SLOT_FREE);
}

/* Stateful causal verifier: rejected drafts mutate both a recurrence and token
 * position. A partially failing commit must restore both; release is counted. */
typedef struct { uint64_t state, saved; size_t position, saved_position; unsigned fail, live, releases; } toy;
static int toy_next(const toy *t) { return (int)((t->state ^ t->position) % 17); }
static void toy_append(toy *t, int token) { t->state = t->state * 6364136223846793005u + (unsigned)token + 1; ++t->position; }
static int toy_snapshot(void *ctx) { toy *t = ctx; if (t->fail == 1) return 1; t->saved = t->state; t->saved_position = t->position; t->live = 1; return 0; }
static int toy_batch(void *ctx, const int *drafts, size_t n, int *rows, size_t rn) {
    toy *t = ctx; REQUIRE(rn == n + 1 && t->live);
    for (size_t i = 0; i <= n; ++i) { rows[i] = toy_next(t); if (i < n) toy_append(t, drafts[i]); if (t->fail == 2) return 1; }
    if (t->fail == 3) rows[n] = 17;
    return 0;
}
static int toy_restore(void *ctx) { toy *t = ctx; REQUIRE(t->live); t->state = t->saved; t->position = t->saved_position; return 0; }
static int toy_commit(void *ctx, const int *tokens, size_t n) { toy *t = ctx; for (size_t i = 0; i < n; ++i) { toy_append(t, tokens[i]); if (t->fail == 4) return 1; } return 0; }
static void toy_release(void *ctx) { toy *t = ctx; REQUIRE(t->live); t->live = 0; ++t->releases; }
static void verify_case(uint64_t *r) {
    toy t = {random64(r), 0, 1, 0, (unsigned)(random64(r) % 5), 0, 0}, original = t, expected = t;
    size_t n = (size_t)(random64(r) % 9), budget = (size_t)(random64(r) % 12);
    int drafts[8], output[9], before[9];
    strata_heads_greedy_ops ops = {&t, toy_snapshot, toy_batch, toy_restore, toy_commit, toy_release, NULL};
    strata_heads_greedy_request req = {drafts, n, budget, 17, 16, 0};
    strata_heads_greedy_result result, old;
    memset(&result, 0x5a, sizeof(result)); old = result;
    for (size_t i = 0; i < 9; ++i) output[i] = before[i] = -123;
    for (size_t i = 0; i < n; ++i) { drafts[i] = toy_next(&expected); toy_append(&expected, drafts[i]); }
    if (n && (random64(r) & 1)) { size_t at = (size_t)(random64(r) % n); drafts[at] = (drafts[at] + 1) % 17; }
    strata_heads_status s = strata_heads_greedy_verify(&ops, &req, output, 9, &result);
    REQUIRE(!t.live);
    if (s != STRATA_HEADS_OK) {
        REQUIRE(!memcmp(output, before, sizeof(output)) && !memcmp(&result, &old, sizeof(result)));
        REQUIRE(t.state == original.state && t.position == original.position);
    } else {
        REQUIRE(result.output_count <= budget && result.output_count <= 9);
        expected = original;
        for (size_t i = 0; i < result.output_count; ++i) { REQUIRE(output[i] == toy_next(&expected)); toy_append(&expected, output[i]); }
        REQUIRE(t.state == expected.state && t.position == expected.position);
        for (size_t i = result.output_count; i < 9; ++i) REQUIRE(output[i] == -123);
    }
    REQUIRE(t.releases == ((!budget || original.fail == 1) ? 0u : 1u));
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    uint64_t r = UINT64_C(1469598103934665603);
    byte_guard arbitrary, saved;
    size_t length = size < FUZZ_BYTES ? size : FUZZ_BYTES;
    for (size_t i = 0; i < length; ++i) r = (r ^ data[i]) * UINT64_C(1099511628211);
    fuzz_case_seed = r;
    byte_guard_init(&arbitrary);
    if (length) memcpy(arbitrary.values, data, length);
    saved = arbitrary;
    check_decode(arbitrary.values, length, FUZZ_VALUES, 0);
    check_decode(arbitrary.values, length, FUZZ_VALUES, 1);
    REQUIRE(!memcmp(&arbitrary, &saved, sizeof(arbitrary)));
    codec_case(&r, 0); codec_case(&r, 1); matvec_case(&r); quant_heads_case(&r);
    prefetch_case(&r); verify_case(&r);
    return 0;
}

#ifndef STRATA_LIBFUZZER
static uint64_t parse_uint(const char *s) {
    char *end; unsigned long long value;
    errno = 0; if (!*s || *s == '-') { fputs("expected unsigned count/seed\n", stderr); exit(2); }
    value = strtoull(s, &end, 0);
    if (errno || *end) { fputs("invalid count/seed\n", stderr); exit(2); }
    return (uint64_t)value;
}
int main(int argc, char **argv) {
    uint64_t cases = argc > 1 ? parse_uint(argv[1]) : 100000;
    uint64_t initial = argc > 2 ? parse_uint(argv[2]) : UINT64_C(20261011), r = initial;
    if (argc > 3 || !cases) { fputs("usage: fuzz_strata_core [cases>0] [seed]\n", stderr); return 2; }
    for (uint64_t i = 0; i < cases; ++i) {
        uint8_t input[64];
        for (size_t j = 0; j < sizeof(input); j += 8) { uint64_t n = random64(&r); for (unsigned k = 0; k < 8; ++k) input[j+k] = (uint8_t)(n >> (8*k)); }
        LLVMFuzzerTestOneInput(input, sizeof(input));
    }
    printf("fuzz_strata_core: %" PRIu64 " deterministic cases passed, initial_seed=%" PRIu64 "\n", cases, initial);
    return 0;
}
#endif

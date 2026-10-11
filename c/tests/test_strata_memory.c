/* Behavioral reference tests; no model weights or GPU are required. */
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include "../strata_memory.h"

static void test_codec(void) {
    enum { N = 257, BLOCK = 32 };
    float input[N], output[N];
    uint8_t encoded[512], snapshot[512], malformed[512];
    size_t need = 0, written = 0, decoded = 0;
    for (size_t i = 0; i < N; ++i)
        input[i] = (float)(sin((double)i * 0.4) * (1 + i % 17));
    assert(coli_strata_kv_size(N, BLOCK, &need) == COLI_STRATA_OK);
    assert(need == 16 + N + 4 * 9 && need < sizeof(input) / 3);
    assert(coli_strata_kv_encode(input, N, BLOCK, encoded, sizeof(encoded), &written)
           == COLI_STRATA_OK && written == need);
    memcpy(snapshot, encoded, written);
    assert(coli_strata_kv_decode(encoded, written, output, N, &decoded)
           == COLI_STRATA_OK && decoded == N);
    assert(memcmp(snapshot, encoded, written) == 0);
    for (size_t start = 0; start < N; start += BLOCK) {
        size_t end = start + BLOCK < N ? start + BLOCK : N;
        double maxabs = 0;
        for (size_t i = start; i < end; ++i)
            if (fabs(input[i]) > maxabs) maxabs = fabs(input[i]);
        for (size_t i = start; i < end; ++i)
            assert(fabs((double)output[i] - input[i]) <= maxabs / 254 + maxabs * FLT_EPSILON);
    }

    /* Zeros, symmetric extrema, and subnormals must not fabricate infinities. */
    {
        float special[] = {0, -0.0f, 0, 0, FLT_MAX, -FLT_MAX,
                           nextafterf(0, 1), -nextafterf(0, 1)};
        float restored[8];
        assert(coli_strata_kv_encode(special, 8, 4, encoded, sizeof(encoded), &written)
               == COLI_STRATA_OK);
        assert(coli_strata_kv_decode(encoded, written, restored, 8, &decoded) == COLI_STRATA_OK);
        for (size_t i = 0; i < 8; ++i) assert(isfinite(restored[i]));
        for (size_t i = 0; i < 4; ++i) assert(restored[i] == 0);
        assert(restored[4] == FLT_MAX && restored[5] == -FLT_MAX);
        /* Use their own block so subnormals are preserved rather than quantized away. */
        assert(coli_strata_kv_encode(special + 6, 2, 2, encoded, sizeof(encoded), &written)
               == COLI_STRATA_OK);
        assert(coli_strata_kv_decode(encoded, written, restored, 8, &decoded) == COLI_STRATA_OK);
        assert(restored[0] == special[6] && restored[1] == special[7]);
    }
    assert(coli_strata_kv_encode(NULL, 0, BLOCK, encoded, sizeof(encoded), &written)
           == COLI_STRATA_OK && written == 16);
    assert(coli_strata_kv_decode(encoded, written, NULL, 0, &decoded)
           == COLI_STRATA_OK && decoded == 0);
    need = 123;
    assert(coli_strata_kv_size(1, 0, &need) == COLI_STRATA_INVALID && need == 123);
    assert(coli_strata_kv_size(1, COLI_STRATA_KV_MAX_BLOCK + 1, &need) == COLI_STRATA_INVALID);
    assert(coli_strata_kv_size(COLI_STRATA_KV_MAX_VALUES, COLI_STRATA_KV_MAX_BLOCK, &need)
           == COLI_STRATA_OK);
    assert(need == 16 + COLI_STRATA_KV_MAX_VALUES + 4 * 4096);
    assert(coli_strata_kv_size(COLI_STRATA_KV_MAX_VALUES + 1, BLOCK, &need) == COLI_STRATA_CAPACITY);
    assert(coli_strata_kv_size(SIZE_MAX, BLOCK, &need) == COLI_STRATA_CAPACITY);
    assert(coli_strata_kv_size(1, BLOCK, NULL) == COLI_STRATA_INVALID);

    /* Failure is transactional: neither the encoded buffer nor result changes. */
    memset(encoded, 0x5a, sizeof(encoded)); memcpy(snapshot, encoded, sizeof(encoded));
    written = 99; input[N - 1] = NAN;
    assert(coli_strata_kv_encode(input, N, BLOCK, encoded, sizeof(encoded), &written)
           == COLI_STRATA_NONFINITE && written == 99);
    assert(memcmp(snapshot, encoded, sizeof(encoded)) == 0);
    input[N - 1] = INFINITY;
    assert(coli_strata_kv_encode(input, N, BLOCK, encoded, sizeof(encoded), &written)
           == COLI_STRATA_NONFINITE);
    input[N - 1] = 1;
    assert(coli_strata_kv_encode(input, N, BLOCK, encoded, 4, &written) == COLI_STRATA_CAPACITY);
    assert(coli_strata_kv_encode(NULL, 1, BLOCK, encoded, sizeof(encoded), &written)
           == COLI_STRATA_INVALID);
    {
        union { float f[16]; uint8_t b[64]; } overlap;
        overlap.f[0] = 1;
        assert(coli_strata_kv_encode(overlap.f, 1, 1, overlap.b, sizeof(overlap), &written)
               == COLI_STRATA_INVALID);
    }

    assert(coli_strata_kv_encode(input, N, BLOCK, encoded, sizeof(encoded), &written)
           == COLI_STRATA_OK);
    decoded = 111; output[0] = 333;
    assert(coli_strata_kv_decode(encoded, written, output, N - 1, &decoded)
           == COLI_STRATA_CAPACITY && output[0] == 333 && decoded == 111);
    assert(coli_strata_kv_decode(encoded, written - 1, output, N, &decoded) == COLI_STRATA_INVALID);
    assert(coli_strata_kv_decode(encoded, 0, output, N, &decoded) == COLI_STRATA_INVALID);
    memcpy(malformed, encoded, written); malformed[0] = 'x';
    assert(coli_strata_kv_decode(malformed, written, output, N, &decoded) == COLI_STRATA_INVALID);
    memcpy(malformed, encoded, written); malformed[12] = 0;
    assert(coli_strata_kv_decode(malformed, written, output, N, &decoded) == COLI_STRATA_INVALID);
    memcpy(malformed, encoded, written); malformed[20] = 128; /* reserved -128 code */
    assert(coli_strata_kv_decode(malformed, written, output, N, &decoded) == COLI_STRATA_INVALID);
    memcpy(malformed, encoded, written); coli_strata_put_u32_(malformed + 16, 0x7fc00000u);
    assert(coli_strata_kv_decode(malformed, written, output, N, &decoded) == COLI_STRATA_INVALID);
    memcpy(malformed, encoded, written); coli_strata_put_u32_(malformed + 16, 0xbf800000u);
    assert(coli_strata_kv_decode(malformed, written, output, N, &decoded) == COLI_STRATA_INVALID);
    memcpy(malformed, encoded, written); coli_strata_put_u32_(malformed + 16, 0); malformed[20] = 1;
    assert(coli_strata_kv_decode(malformed, written, output, N, &decoded) == COLI_STRATA_INVALID);
    assert(output[0] == 333 && decoded == 111);
}

static void test_hot_policy(void) {
    coli_strata_hot_entry entries[6];
    coli_strata_hot_policy policy;
    coli_strata_hot_rank scratch[6];
    size_t selected[6], count = 0, bytes = 0;
    assert(coli_strata_hot_init(&policy, entries, 6, 2, 3, 0.5) == COLI_STRATA_OK);
    for (uint32_t l = 0; l < 2; ++l)
        for (uint32_t e = 0; e < 3; ++e)
            assert(coli_strata_hot_configure(&policy, l, e, 10, COLI_STRATA_ABSENT, 0)
                   == COLI_STRATA_OK);
    assert(coli_strata_hot_touch(&policy, 0, 0, 0, 64) == COLI_STRATA_OK);
    assert(coli_strata_hot_touch(&policy, 1, 1, 0, 32) == COLI_STRATA_OK);
    assert(coli_strata_hot_select(&policy, 0, 20, scratch, 6, selected, 6, &count, &bytes)
           == COLI_STRATA_OK && count == 2 && bytes == 20);
    assert(selected[0] == 0 && selected[1] == 4);

    /* New workload replaces old popularity through explicit epoch decay. */
    assert(coli_strata_hot_touch(&policy, 0, 2, 10, 2) == COLI_STRATA_OK);
    assert(coli_strata_hot_select(&policy, 10, 10, scratch, 6, selected, 6, &count, &bytes)
           == COLI_STRATA_OK && count == 1 && selected[0] == 2);
    assert(coli_strata_hot_configure(&policy, 1, 2, 10, COLI_STRATA_RESIDENT, 1)
           == COLI_STRATA_OK);
    assert(coli_strata_hot_select(&policy, 10, 20, scratch, 6, selected, 6, &count, &bytes)
           == COLI_STRATA_OK && count == 2 && selected[0] == 5 && selected[1] == 2);
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1; j < count; ++j) assert(selected[i] != selected[j]);
    selected[0] = 99; count = 99; bytes = 99;
    assert(coli_strata_hot_select(&policy, 10, 9, scratch, 6, selected, 6, &count, &bytes)
           == COLI_STRATA_CAPACITY && selected[0] == 99 && count == 99 && bytes == 99);
    assert(coli_strata_hot_select(&policy, 10, 20, scratch, 6, selected, 1, &count, &bytes)
           == COLI_STRATA_CAPACITY && selected[0] == 99);
    assert(coli_strata_hot_select(&policy, 10, 20, scratch, 5, selected, 6, &count, &bytes)
           == COLI_STRATA_CAPACITY);
    assert(coli_strata_hot_select(&policy, 10, 20, scratch, 6, selected, 6, &count, &count)
           == COLI_STRATA_INVALID);
    assert(coli_strata_hot_select(&policy, 10, 20, scratch, 6, selected, 6, selected, &bytes)
           == COLI_STRATA_INVALID);
    assert(coli_strata_hot_select(&policy, 10, 20, scratch, 6, selected, SIZE_MAX, &count, &bytes)
           == COLI_STRATA_OVERFLOW);
    assert(coli_strata_hot_touch(&policy, 2, 0, 10, 1) == COLI_STRATA_INVALID);
    assert(coli_strata_hot_touch(&policy, 0, 3, 10, 1) == COLI_STRATA_INVALID);
    assert(coli_strata_hot_touch(&policy, 0, 0, 9, 1) == COLI_STRATA_INVALID);
    assert(coli_strata_hot_touch(&policy, 0, 0, 10, NAN) == COLI_STRATA_INVALID);
    assert(coli_strata_hot_touch(&policy, 0, 0, 10, -1) == COLI_STRATA_INVALID);
    assert(coli_strata_hot_touch(&policy, 0, 0, 10, DBL_MAX) == COLI_STRATA_OK);
    assert(coli_strata_hot_touch(&policy, 0, 0, 10, DBL_MAX) == COLI_STRATA_OVERFLOW);
    assert(coli_strata_hot_configure(&policy, 0, 0, 0, COLI_STRATA_RESIDENT, 1)
           == COLI_STRATA_INVALID);

    /* Ties have deterministic residency hysteresis, then stable identity. */
    assert(coli_strata_hot_init(&policy, entries, 6, 1, 3, 0.5) == COLI_STRATA_OK);
    assert(coli_strata_hot_configure(&policy, 0, 0, 10, COLI_STRATA_ABSENT, 0) == COLI_STRATA_OK);
    assert(coli_strata_hot_configure(&policy, 0, 1, 10, COLI_STRATA_WARMING, 0) == COLI_STRATA_OK);
    assert(coli_strata_hot_configure(&policy, 0, 2, 10, COLI_STRATA_RESIDENT, 0) == COLI_STRATA_OK);
    assert(coli_strata_hot_select(&policy, UINT64_MAX, 20, scratch, 6, selected, 6, &count, &bytes)
           == COLI_STRATA_OK && selected[0] == 2 && selected[1] == 1);
    assert(coli_strata_hot_select(&policy, 0, 0, scratch, 6, NULL, 0, &count, &bytes)
           == COLI_STRATA_OK && count == 0 && bytes == 0);
    assert(coli_strata_hot_configure(&policy, 0, 0, SIZE_MAX, COLI_STRATA_ABSENT, 1)
           == COLI_STRATA_OK);
    assert(coli_strata_hot_configure(&policy, 0, 1, 1, COLI_STRATA_ABSENT, 1) == COLI_STRATA_OK);
    assert(coli_strata_hot_select(&policy, 0, SIZE_MAX, scratch, 6, selected, 6, &count, &bytes)
           == COLI_STRATA_CAPACITY); /* sum must not wrap */
    assert(coli_strata_hot_init(&policy, entries, 6, UINT32_MAX, UINT32_MAX, 0.5)
           != COLI_STRATA_OK);
    assert(coli_strata_hot_init(&policy, entries, 6, 1, 3, 0) == COLI_STRATA_INVALID);
}

typedef struct {
    int transfer[32], consumer[32], reject_enqueue, reject_consume;
    unsigned submissions, consumes;
    uint64_t next_transfer, next_consumer;
} fake_backend;

static int fake_enqueue(void *ctx, unsigned slot,
        const coli_strata_prefetch_identity *id, uint64_t *event) {
    fake_backend *b = ctx;
    assert(slot < 2 && id->bytes <= 64);
    if (b->reject_enqueue) return 0;
    assert(b->next_transfer < 32);
    *event = b->next_transfer++;
    ++b->submissions;
    return 1;
}
static int fake_transfer_status(void *ctx, uint64_t event) {
    fake_backend *b = ctx;
    assert(event < b->next_transfer);
    return b->transfer[event];
}
static int fake_consume(void *ctx, unsigned slot,
        const coli_strata_prefetch_identity *id, uint64_t *event) {
    fake_backend *b = ctx;
    assert(slot < 2 && id->bytes <= 64);
    if (b->reject_consume) return 0;
    assert(b->next_consumer < 32);
    *event = b->next_consumer++;
    ++b->consumes;
    return 1;
}
static int fake_consumer_status(void *ctx, uint64_t event) {
    fake_backend *b = ctx;
    assert(event < b->next_consumer);
    return b->consumer[event];
}

static void test_prefetch(void) {
    fake_backend fake = {0};
    coli_strata_prefetch_backend backend = {&fake, fake_enqueue, fake_transfer_status,
                                           fake_consume, fake_consumer_status};
    coli_strata_prefetch pool;
    coli_strata_prefetch_ticket a, b, acquired, c;
    coli_strata_prefetch_identity first = {0, 1, 10, 64}, second = {1, 2, 11, 64};
    coli_strata_prefetch_identity demand = {1, 3, 11, 64};
    assert(coli_strata_prefetch_init(&pool, 3, 4, 64, 64, &backend) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_enqueue(&pool, &first, (coli_strata_prefetch_ticket *)&pool)
           == COLI_STRATA_INVALID);
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_enqueue(&pool, &second, &b) == COLI_STRATA_OK && a.slot != b.slot);
    assert(coli_strata_prefetch_enqueue(&pool, &demand, &c) == COLI_STRATA_BUSY);
    assert(coli_strata_prefetch_enqueue(&pool, &first, &c) == COLI_STRATA_BUSY);
    assert(fake.submissions == 2);
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_PENDING);
    assert(coli_strata_prefetch_acquire(&pool, &first, &acquired) == COLI_STRATA_PENDING);
    assert(coli_strata_prefetch_acquire(&pool, &demand, &acquired) == COLI_STRATA_MISS);
    assert(fake.consumes == 0); /* guessed route must never replace actual routing */

    fake.transfer[0] = 1;
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_PENDING);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_READY);
    assert(coli_strata_prefetch_acquire(&pool, &first, (coli_strata_prefetch_ticket *)&pool)
           == COLI_STRATA_INVALID);
    fake.reject_consume = 1;
    assert(coli_strata_prefetch_acquire(&pool, &first, &acquired) == COLI_STRATA_BACKEND_ERROR);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_READY);
    fake.reject_consume = 0;
    assert(coli_strata_prefetch_acquire(&pool, &first, &acquired) == COLI_STRATA_OK);
    assert(acquired.slot == a.slot && acquired.generation == a.generation);
    assert(coli_strata_prefetch_acquire(&pool, &first, &c) == COLI_STRATA_BUSY);
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_PENDING);
    assert(coli_strata_prefetch_enqueue(&pool, &demand, &c) == COLI_STRATA_BUSY);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_IN_USE);
    fake.consumer[0] = 1;
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_OK);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_FREE);
    assert(coli_strata_prefetch_enqueue(&pool, &demand, &c) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_cancel(&pool, a) == COLI_STRATA_INVALID); /* stale ticket */

    /* Discard a wrong prediction only after its asynchronous transfer drains. */
    assert(coli_strata_prefetch_cancel(&pool, b) == COLI_STRATA_PENDING);
    assert(pool.slots[b.slot].state == COLI_STRATA_SLOT_IN_FLIGHT);
    assert(coli_strata_prefetch_acquire(&pool, &second, &acquired) == COLI_STRATA_CANCELLED);
    assert(coli_strata_prefetch_enqueue(&pool, &first, &acquired) == COLI_STRATA_BUSY);
    fake.transfer[1] = 1;
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_PENDING);
    assert(pool.slots[b.slot].state == COLI_STRATA_SLOT_FREE);
    fake.transfer[2] = 1;
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_acquire(&pool, &demand, &acquired) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_cancel(&pool, acquired) == COLI_STRATA_PENDING);
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_PENDING);
    fake.consumer[1] = 1;
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_CANCELLED);

    /* Enqueue/transfer/consumer errors permit fallback without early reuse. */
    fake.reject_enqueue = 1;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_BACKEND_ERROR);
    assert(pool.slots[0].state == COLI_STRATA_SLOT_FREE &&
           pool.slots[1].state == COLI_STRATA_SLOT_FREE);
    fake.reject_enqueue = 0;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_OK);
    fake.transfer[3] = 2; /* invalid backend status cannot be trusted as a fence */
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_BACKEND_ERROR);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_IN_FLIGHT);
    fake.transfer[3] = -1;
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_BACKEND_ERROR);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_FREE);
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_OK);
    fake.transfer[4] = 1;
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_cancel(&pool, a) == COLI_STRATA_CANCELLED);
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_OK);
    fake.transfer[5] = 1;
    assert(coli_strata_prefetch_poll(&pool) == COLI_STRATA_OK);
    assert(coli_strata_prefetch_acquire(&pool, &first, &acquired) == COLI_STRATA_OK);
    fake.consumer[2] = 2;
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_BACKEND_ERROR);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_IN_USE);
    fake.consumer[2] = -1;
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_BACKEND_ERROR);
    assert(pool.slots[a.slot].state == COLI_STRATA_SLOT_FREE);
    assert(coli_strata_prefetch_release(&pool, acquired) == COLI_STRATA_INVALID);

    first.layer = 3;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_INVALID);
    first.layer = 0; first.expert = 4;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_INVALID);
    first.expert = 1; first.bytes = 0;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_INVALID);
    first.bytes = 65;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_CAPACITY);
    first.bytes = 64; pool.generation = UINT64_MAX;
    assert(coli_strata_prefetch_enqueue(&pool, &first, &a) == COLI_STRATA_OVERFLOW);
}

int main(void) {
    test_codec();
    test_hot_policy();
    test_prefetch();
    puts("test_strata_memory: ok");
    return 0;
}

/* Experimental, portable Strata-inspired reference primitives.
 *
 * These routines do not integrate with a model's KV cache, router, expert
 * store, CUDA streams, or PCIe transfers. The INT8 codec is LOSSY and needs
 * model-specific quality evaluation before use. The hot policy selects
 * residency candidates only: it never changes weights or routing, allocates
 * device memory, locks pages, or establishes residency. The caller owns all
 * buffers and must serialize calls, including callbacks (no reentrancy).
 *
 * All storage is caller-owned. Apart from the C standard library there are
 * no dependencies. Decode never changes the encoded bytes. Array arguments
 * must refer to actual objects of the stated capacity; no C API can verify
 * an invented capacity. Structures should be changed only through this API.
 */
#ifndef COLI_STRATA_MEMORY_H
#define COLI_STRATA_MEMORY_H

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Strata memory reference requires finite checks; do not use fast-math"
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

typedef enum {
    COLI_STRATA_OK = 0,
    COLI_STRATA_INVALID,
    COLI_STRATA_CAPACITY,
    COLI_STRATA_OVERFLOW,
    COLI_STRATA_NONFINITE,
    COLI_STRATA_BUSY,
    COLI_STRATA_PENDING,
    COLI_STRATA_MISS,
    COLI_STRATA_CANCELLED,
    COLI_STRATA_BACKEND_ERROR
} coli_strata_status;

#define COLI_STRATA_KV_MAX_VALUES ((size_t)16777216)
#define COLI_STRATA_KV_MAX_BLOCK ((size_t)4096)
#define COLI_STRATA_HOT_MAX_ENTRIES ((size_t)1048576)
#define COLI_STRATA_KV_HEADER_BYTES ((size_t)16)

static inline int coli_strata_overlap_(const void *a, size_t an,
                                       const void *b, size_t bn) {
    uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
    if (!an || !bn) return 0;
    if (an > UINTPTR_MAX - aa || bn > UINTPTR_MAX - bb) return 1;
    return aa < bb + bn && bb < aa + an;
}

static inline void coli_strata_put_u32_(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static inline uint32_t coli_strata_get_u32_(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Wire format: "CSK1", u32 value count, u32 block length, u32 block count;
 * then each block's IEEE binary32 maxabs followed by its signed INT8 codes.
 * Integers/float bits are little endian. Codes are in [-127,127]. The scale
 * is maxabs/127, calculated in double so even float subnormals are supported.
 * Saving maxabs instead of a rounded float scale also supports FLT_MAX.
 * Nonzero blocks have <= maxabs/254 rounding error, plus float rounding.
 * Output/result parameters are modified only on success. Buffers/results
 * must be disjoint. This format is a reference vector codec, not a model KV
 * layout or a claim that a model's existing attention kernel accepts INT8.
 */
static inline coli_strata_status coli_strata_kv_size(size_t count,
        size_t block_values, size_t *bytes) {
    size_t blocks, need;
    if (!bytes || !block_values || block_values > COLI_STRATA_KV_MAX_BLOCK)
        return COLI_STRATA_INVALID;
    if (count > COLI_STRATA_KV_MAX_VALUES) return COLI_STRATA_CAPACITY;
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
        FLT_MAX_EXP != 128) return COLI_STRATA_INVALID;
    blocks = count / block_values + (count % block_values != 0);
    if (blocks > (SIZE_MAX - COLI_STRATA_KV_HEADER_BYTES) / 4)
        return COLI_STRATA_OVERFLOW;
    need = COLI_STRATA_KV_HEADER_BYTES + blocks * 4;
    if (count > SIZE_MAX - need) return COLI_STRATA_OVERFLOW;
    *bytes = need + count;
    return COLI_STRATA_OK;
}

static inline coli_strata_status coli_strata_kv_encode(const float *src,
        size_t count, size_t block_values, uint8_t *encoded,
        size_t capacity, size_t *written) {
    size_t need, src_bytes, offset = COLI_STRATA_KV_HEADER_BYTES;
    coli_strata_status s = coli_strata_kv_size(count, block_values, &need);
    if (s != COLI_STRATA_OK) return s;
    if (!encoded || !written || (count && !src)) return COLI_STRATA_INVALID;
    if (capacity < need) return COLI_STRATA_CAPACITY;
    if (count > SIZE_MAX / sizeof(float)) return COLI_STRATA_OVERFLOW;
    src_bytes = count * sizeof(float);
    if (coli_strata_overlap_(src, src_bytes, encoded, need) ||
        coli_strata_overlap_(written, sizeof(*written), encoded, need) ||
        coli_strata_overlap_(written, sizeof(*written), src, src_bytes))
        return COLI_STRATA_INVALID;
    /* Validate before writing anything, including the header. */
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(src[i])) return COLI_STRATA_NONFINITE;
    memcpy(encoded, "CSK1", 4);
    coli_strata_put_u32_(encoded + 4, (uint32_t)count);
    coli_strata_put_u32_(encoded + 8, (uint32_t)block_values);
    coli_strata_put_u32_(encoded + 12,
            (uint32_t)(count / block_values + (count % block_values != 0)));
    for (size_t start = 0; start < count;) {
        size_t n = count - start;
        float maxabs = 0.0f;
        uint32_t bits;
        if (n > block_values) n = block_values;
        for (size_t j = 0; j < n; ++j) {
            float a = fabsf(src[start + j]);
            if (a > maxabs) maxabs = a;
        }
        memcpy(&bits, &maxabs, sizeof(bits));
        coli_strata_put_u32_(encoded + offset, bits);
        offset += 4;
        for (size_t j = 0; j < n; ++j) {
            double x = maxabs ? ((double)src[start + j] / maxabs) * 127.0 : 0;
            int q = (int)(x + (x < 0 ? -0.5 : 0.5));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            encoded[offset++] = (uint8_t)(q < 0 ? 256 + q : q);
        }
        start += n;
    }
    *written = need;
    return COLI_STRATA_OK;
}

static inline coli_strata_status coli_strata_kv_decode(const uint8_t *encoded,
        size_t encoded_bytes, float *dst, size_t capacity, size_t *decoded) {
    size_t count, block_values, blocks, need, offset;
    coli_strata_status s;
    if (!encoded || !decoded || encoded_bytes < COLI_STRATA_KV_HEADER_BYTES)
        return COLI_STRATA_INVALID;
    if (memcmp(encoded, "CSK1", 4)) return COLI_STRATA_INVALID;
    count = coli_strata_get_u32_(encoded + 4);
    block_values = coli_strata_get_u32_(encoded + 8);
    blocks = coli_strata_get_u32_(encoded + 12);
    s = coli_strata_kv_size(count, block_values, &need);
    if (s != COLI_STRATA_OK) return s;
    if (blocks != count / block_values + (count % block_values != 0) ||
        encoded_bytes != need) return COLI_STRATA_INVALID;
    if (capacity < count) return COLI_STRATA_CAPACITY;
    if (count && !dst) return COLI_STRATA_INVALID;
    if (count > SIZE_MAX / sizeof(float)) return COLI_STRATA_OVERFLOW;
    if (coli_strata_overlap_(encoded, need, dst, count * sizeof(float)) ||
        coli_strata_overlap_(decoded, sizeof(*decoded), encoded, need) ||
        coli_strata_overlap_(decoded, sizeof(*decoded), dst, count * sizeof(float)))
        return COLI_STRATA_INVALID;
    /* Reject malformed scales/codes before touching the output. */
    offset = COLI_STRATA_KV_HEADER_BYTES;
    for (size_t start = 0; start < count;) {
        size_t n = count - start;
        uint32_t bits = coli_strata_get_u32_(encoded + offset);
        float maxabs;
        memcpy(&maxabs, &bits, sizeof(maxabs));
        if (!isfinite(maxabs) || maxabs < 0) return COLI_STRATA_INVALID;
        if (n > block_values) n = block_values;
        offset += 4;
        for (size_t j = 0; j < n; ++j)
            if (encoded[offset + j] == 128 ||
                (!maxabs && encoded[offset + j])) return COLI_STRATA_INVALID;
        offset += n;
        start += n;
    }
    offset = COLI_STRATA_KV_HEADER_BYTES;
    for (size_t start = 0; start < count;) {
        size_t n = count - start;
        uint32_t bits = coli_strata_get_u32_(encoded + offset);
        float maxabs;
        memcpy(&maxabs, &bits, sizeof(maxabs));
        if (n > block_values) n = block_values;
        offset += 4;
        for (size_t j = 0; j < n; ++j) {
            int q = encoded[offset++];
            if (q > 127) q -= 256;
            dst[start + j] = (float)(q * ((double)maxabs / 127.0));
        }
        start += n;
    }
    *decoded = count;
    return COLI_STRATA_OK;
}

/* Dense caller-owned history; a zero byte size disables an entry. Families
 * with unequal expert counts can leave padding entries disabled. Identity is
 * index/layer/expert, so selection cannot contain duplicate experts. A pin is
 * a policy reservation, NOT OS mlock or a device residency guarantee.
 */
typedef enum {
    COLI_STRATA_ABSENT = 0,
    COLI_STRATA_WARMING = 1,
    COLI_STRATA_RESIDENT = 2
} coli_strata_residency;

typedef struct {
    size_t bytes;
    double heat;
    uint64_t touched_at;
    unsigned char residency, pinned;
} coli_strata_hot_entry;

typedef struct {
    coli_strata_hot_entry *entries;
    size_t count;
    uint32_t layers, experts_per_layer;
    uint64_t epoch;
    double decay;
} coli_strata_hot_policy;

typedef struct {
    size_t index;
    double score;
    unsigned char residency, pinned;
} coli_strata_hot_rank;

static inline int coli_strata_hot_valid_(const coli_strata_hot_policy *p) {
    return p && p->entries && p->layers && p->experts_per_layer &&
           p->count <= COLI_STRATA_HOT_MAX_ENTRIES &&
           p->layers <= SIZE_MAX / p->experts_per_layer &&
           p->count == (size_t)p->layers * p->experts_per_layer &&
           isfinite(p->decay) && p->decay > 0 && p->decay <= 1;
}

static inline coli_strata_status coli_strata_hot_init(coli_strata_hot_policy *p,
        coli_strata_hot_entry *entries, size_t capacity, uint32_t layers,
        uint32_t experts_per_layer, double decay) {
    size_t n;
    if (!p || !entries || !layers || !experts_per_layer ||
        !isfinite(decay) || decay <= 0 || decay > 1) return COLI_STRATA_INVALID;
    if (layers > SIZE_MAX / experts_per_layer) return COLI_STRATA_OVERFLOW;
    n = (size_t)layers * experts_per_layer;
    if (n > capacity || n > COLI_STRATA_HOT_MAX_ENTRIES) return COLI_STRATA_CAPACITY;
    if (n > SIZE_MAX / sizeof(*entries)) return COLI_STRATA_OVERFLOW;
    if (coli_strata_overlap_(p, sizeof(*p), entries, n * sizeof(*entries)))
        return COLI_STRATA_INVALID;
    memset(entries, 0, n * sizeof(*entries));
    p->entries = entries; p->count = n; p->layers = layers;
    p->experts_per_layer = experts_per_layer; p->epoch = 0; p->decay = decay;
    return COLI_STRATA_OK;
}

static inline coli_strata_status coli_strata_hot_configure(coli_strata_hot_policy *p,
        uint32_t layer, uint32_t expert, size_t bytes,
        coli_strata_residency residency, int pinned) {
    coli_strata_hot_entry *e;
    if (!coli_strata_hot_valid_(p) || layer >= p->layers ||
        expert >= p->experts_per_layer || residency < COLI_STRATA_ABSENT ||
        residency > COLI_STRATA_RESIDENT || (pinned != 0 && pinned != 1) ||
        (!bytes && (pinned || residency != COLI_STRATA_ABSENT)))
        return COLI_STRATA_INVALID;
    e = &p->entries[(size_t)layer * p->experts_per_layer + expert];
    e->bytes = bytes; e->residency = (unsigned char)residency;
    e->pinned = (unsigned char)pinned;
    if (!bytes) { e->heat = 0; e->touched_at = p->epoch; }
    return COLI_STRATA_OK;
}

static inline coli_strata_status coli_strata_hot_touch(coli_strata_hot_policy *p,
        uint32_t layer, uint32_t expert, uint64_t epoch, double weight) {
    coli_strata_hot_entry *e;
    double score, new_heat;
    if (!coli_strata_hot_valid_(p) || layer >= p->layers ||
        expert >= p->experts_per_layer || epoch < p->epoch ||
        !isfinite(weight) || weight < 0) return COLI_STRATA_INVALID;
    e = &p->entries[(size_t)layer * p->experts_per_layer + expert];
    if (!e->bytes || !isfinite(e->heat) || e->heat < 0 || e->touched_at > epoch)
        return COLI_STRATA_INVALID;
    score = e->heat * pow(p->decay, (double)(epoch - e->touched_at));
    if (weight > DBL_MAX - score) return COLI_STRATA_OVERFLOW;
    new_heat = score + weight;
    if (!isfinite(new_heat)) return COLI_STRATA_OVERFLOW;
    e->heat = new_heat; e->touched_at = epoch; p->epoch = epoch;
    return COLI_STRATA_OK;
}

static inline int coli_strata_hot_compare_(const void *av, const void *bv) {
    const coli_strata_hot_rank *a = (const coli_strata_hot_rank *)av;
    const coli_strata_hot_rank *b = (const coli_strata_hot_rank *)bv;
    if (a->pinned != b->pinned) return a->pinned ? -1 : 1;
    if (a->score != b->score) return a->score > b->score ? -1 : 1;
    if (a->residency != b->residency) return a->residency > b->residency ? -1 : 1;
    return a->index < b->index ? -1 : a->index != b->index;
}

/* O(n log n) ranking with scratch supplied by the caller, then greedy
 * admission by decayed heat. This is not an optimal knapsack solver.
 * Equal scores prefer resident, then warming, then smaller indices, reducing
 * unnecessary churn. All pinned entries must fit or CAPACITY is returned.
 * History/residency and output remain unchanged on failure; scratch may be
 * changed. The backend must apply selections with its own lifetime fences.
 */
static inline coli_strata_status coli_strata_hot_select(const coli_strata_hot_policy *p,
        uint64_t epoch, size_t byte_budget, coli_strata_hot_rank *scratch,
        size_t scratch_capacity, size_t *selected, size_t selected_capacity,
        size_t *selected_count, size_t *selected_bytes) {
    size_t n = 0, pinned_bytes = 0, used = 0, count = 0;
    if (!coli_strata_hot_valid_(p) || epoch < p->epoch || !scratch ||
        !selected_count || !selected_bytes || (!selected && selected_capacity))
        return COLI_STRATA_INVALID;
    if (scratch_capacity < p->count) return COLI_STRATA_CAPACITY;
    if (p->count > SIZE_MAX / sizeof(*scratch) ||
        p->count > SIZE_MAX / sizeof(*p->entries) ||
        selected_capacity > SIZE_MAX / sizeof(*selected)) return COLI_STRATA_OVERFLOW;
    {
        const void *buffers[] = {p, p->entries, scratch, selected, selected_count, selected_bytes};
        size_t sizes[] = {sizeof(*p), p->count * sizeof(*p->entries),
                         p->count * sizeof(*scratch), selected_capacity * sizeof(*selected),
                         sizeof(*selected_count), sizeof(*selected_bytes)};
        /* Writable scratch, selections, and results may not alias history,
         * policy, or one another (even on failure paths). */
        for (size_t i = 2; i < 6; ++i)
            for (size_t j = 0; j < i; ++j)
                if (coli_strata_overlap_(buffers[i], sizes[i], buffers[j], sizes[j]))
                    return COLI_STRATA_INVALID;
    }
    /* Validate all histories and pinned budget before filling scratch. */
    for (size_t i = 0; i < p->count; ++i) {
        const coli_strata_hot_entry *e = &p->entries[i];
        if (!isfinite(e->heat) || e->heat < 0 || e->touched_at > epoch ||
            e->residency > COLI_STRATA_RESIDENT || e->pinned > 1 ||
            (!e->bytes && (e->pinned || e->residency))) return COLI_STRATA_INVALID;
        if (e->pinned) {
            if (e->bytes > byte_budget - pinned_bytes) return COLI_STRATA_CAPACITY;
            pinned_bytes += e->bytes;
        }
    }
    for (size_t i = 0; i < p->count; ++i) {
        const coli_strata_hot_entry *e = &p->entries[i];
        if (!e->bytes) continue;
        scratch[n].index = i;
        scratch[n].score = e->heat * pow(p->decay, (double)(epoch - e->touched_at));
        scratch[n].residency = e->residency; scratch[n].pinned = e->pinned;
        ++n;
    }
    qsort(scratch, n, sizeof(*scratch), coli_strata_hot_compare_);
    for (size_t i = 0; i < n; ++i) {
        size_t bytes = p->entries[scratch[i].index].bytes;
        if (bytes <= byte_budget - used) { used += bytes; ++count; }
    }
    if (count > selected_capacity) return COLI_STRATA_CAPACITY;
    used = 0; count = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t bytes = p->entries[scratch[i].index].bytes;
        if (bytes <= byte_budget - used) {
            used += bytes; selected[count++] = scratch[i].index;
        }
    }
    *selected_count = count; *selected_bytes = used;
    return COLI_STRATA_OK;
}

/* Two-slot prefetch reference lifecycle. Backend callbacks return 1 when
 * work was submitted, 0 when rejected with NO work outstanding. Event status
 * callbacks return 0 pending, 1 completed successfully, -1 completed with
 * failure. A terminal result MUST mean the backend no longer accesses the
 * slot's buffer. Events must identify the actual submission, not a reusable
 * slot number. Consumer submission follows the same contract as transfer.
 * No actual transfer, cancellation, parallelism, or GPU overlap is provided.
 * Cancelling means discard on completion, never recycling an active buffer.
 */
typedef struct {
    uint32_t layer, expert;
    uint64_t tag; /* caller's request/step identity; prevents stale predictions */
    size_t bytes;
} coli_strata_prefetch_identity;

typedef struct {
    void *context;
    int (*enqueue)(void *, unsigned, const coli_strata_prefetch_identity *, uint64_t *);
    int (*transfer_status)(void *, uint64_t);
    int (*consume)(void *, unsigned, const coli_strata_prefetch_identity *, uint64_t *);
    int (*consumer_status)(void *, uint64_t);
} coli_strata_prefetch_backend;

typedef enum {
    COLI_STRATA_SLOT_FREE = 0,
    COLI_STRATA_SLOT_IN_FLIGHT,
    COLI_STRATA_SLOT_READY,
    COLI_STRATA_SLOT_IN_USE
} coli_strata_slot_state;

typedef struct {
    unsigned slot;
    uint64_t generation;
} coli_strata_prefetch_ticket;

typedef struct {
    coli_strata_slot_state state;
    coli_strata_prefetch_identity identity;
    uint64_t transfer_event, consumer_event, generation;
    size_t capacity;
    unsigned char discard;
} coli_strata_prefetch_slot;

typedef struct {
    coli_strata_prefetch_slot slots[2];
    coli_strata_prefetch_backend backend;
    uint32_t layers, experts_per_layer;
    uint64_t generation;
    unsigned next_slot;
} coli_strata_prefetch;

static inline int coli_strata_prefetch_valid_(const coli_strata_prefetch *p) {
    return p && p->layers && p->experts_per_layer && p->next_slot < 2 &&
           p->slots[0].capacity && p->slots[1].capacity &&
           p->backend.enqueue && p->backend.transfer_status &&
           p->backend.consume && p->backend.consumer_status;
}

static inline int coli_strata_prefetch_id_valid_(const coli_strata_prefetch *p,
        const coli_strata_prefetch_identity *id) {
    return id && id->layer < p->layers && id->expert < p->experts_per_layer && id->bytes;
}

static inline int coli_strata_prefetch_equal_(const coli_strata_prefetch_identity *a,
        const coli_strata_prefetch_identity *b) {
    return a->layer == b->layer && a->expert == b->expert &&
           a->tag == b->tag && a->bytes == b->bytes;
}

static inline void coli_strata_prefetch_free_(coli_strata_prefetch_slot *s) {
    size_t capacity = s->capacity;
    memset(s, 0, sizeof(*s)); s->capacity = capacity;
}

static inline coli_strata_status coli_strata_prefetch_init(coli_strata_prefetch *p,
        uint32_t layers, uint32_t experts_per_layer, size_t slot0_capacity,
        size_t slot1_capacity, const coli_strata_prefetch_backend *backend) {
    coli_strata_prefetch_backend copy;
    if (!p || !layers || !experts_per_layer || !slot0_capacity ||
        !slot1_capacity || !backend || !backend->enqueue ||
        !backend->transfer_status || !backend->consume || !backend->consumer_status)
        return COLI_STRATA_INVALID;
    /* Init is for new/drained pools only. Reinitializing active work is unsafe. */
    copy = *backend; memset(p, 0, sizeof(*p)); p->backend = copy;
    p->layers = layers; p->experts_per_layer = experts_per_layer;
    p->slots[0].capacity = slot0_capacity; p->slots[1].capacity = slot1_capacity;
    return COLI_STRATA_OK;
}

static inline coli_strata_status coli_strata_prefetch_enqueue(coli_strata_prefetch *p,
        const coli_strata_prefetch_identity *id, coli_strata_prefetch_ticket *ticket) {
    unsigned slot = 2;
    uint64_t event = 0;
    coli_strata_prefetch_identity copy;
    if (!coli_strata_prefetch_valid_(p) || !ticket ||
        !coli_strata_prefetch_id_valid_(p, id)) return COLI_STRATA_INVALID;
    if (coli_strata_overlap_(ticket, sizeof(*ticket), p, sizeof(*p)))
        return COLI_STRATA_INVALID;
    if (p->generation == UINT64_MAX) return COLI_STRATA_OVERFLOW;
    if (id->bytes > p->slots[0].capacity && id->bytes > p->slots[1].capacity)
        return COLI_STRATA_CAPACITY;
    for (unsigned i = 0; i < 2; ++i)
        if (p->slots[i].state != COLI_STRATA_SLOT_FREE &&
            coli_strata_prefetch_equal_(&p->slots[i].identity, id)) return COLI_STRATA_BUSY;
    for (unsigned i = 0; i < 2; ++i) {
        unsigned candidate = (p->next_slot + i) % 2;
        if (p->slots[candidate].state == COLI_STRATA_SLOT_FREE &&
            id->bytes <= p->slots[candidate].capacity) { slot = candidate; break; }
    }
    if (slot == 2) return COLI_STRATA_BUSY;
    copy = *id;
    if (p->backend.enqueue(p->backend.context, slot, &copy, &event) != 1)
        return COLI_STRATA_BACKEND_ERROR;
    p->slots[slot].identity = copy; p->slots[slot].transfer_event = event;
    p->slots[slot].generation = ++p->generation;
    p->slots[slot].state = COLI_STRATA_SLOT_IN_FLIGHT;
    p->next_slot = (slot + 1) % 2;
    ticket->slot = slot; ticket->generation = p->generation;
    return COLI_STRATA_OK;
}

/* Poll advances transfers only. IN_USE slots require release, which checks
 * the consumer completion event. Cancellation never skips either fence.
 */
static inline coli_strata_status coli_strata_prefetch_poll(coli_strata_prefetch *p) {
    int pending = 0, cancelled = 0, failed = 0;
    if (!coli_strata_prefetch_valid_(p)) return COLI_STRATA_INVALID;
    for (unsigned i = 0; i < 2; ++i) {
        coli_strata_prefetch_slot *s = &p->slots[i];
        int result;
        if (s->state != COLI_STRATA_SLOT_IN_FLIGHT) continue;
        result = p->backend.transfer_status(p->backend.context, s->transfer_event);
        if (result == 0) { pending = 1; continue; }
        if (result != 1 && result != -1) { failed = 1; continue; }
        if (result == -1) { failed = 1; coli_strata_prefetch_free_(s); }
        else if (s->discard) { cancelled = 1; coli_strata_prefetch_free_(s); }
        else s->state = COLI_STRATA_SLOT_READY;
    }
    if (failed) return COLI_STRATA_BACKEND_ERROR;
    if (pending) return COLI_STRATA_PENDING;
    return cancelled ? COLI_STRATA_CANCELLED : COLI_STRATA_OK;
}

/* Actual routing identity must match every field before consumption. MISS
 * leaves unrelated predictions intact; cancel them to discard/drain safely
 * and use the caller's synchronous demand path for the unpredicted expert.
 */
static inline coli_strata_status coli_strata_prefetch_acquire(coli_strata_prefetch *p,
        const coli_strata_prefetch_identity *actual, coli_strata_prefetch_ticket *ticket) {
    if (!coli_strata_prefetch_valid_(p) || !ticket ||
        !coli_strata_prefetch_id_valid_(p, actual)) return COLI_STRATA_INVALID;
    if (coli_strata_overlap_(ticket, sizeof(*ticket), p, sizeof(*p)))
        return COLI_STRATA_INVALID;
    for (unsigned i = 0; i < 2; ++i) {
        coli_strata_prefetch_slot *s = &p->slots[i];
        uint64_t event = 0;
        if (s->state == COLI_STRATA_SLOT_FREE ||
            !coli_strata_prefetch_equal_(&s->identity, actual)) continue;
        if (s->discard) return COLI_STRATA_CANCELLED;
        if (s->state == COLI_STRATA_SLOT_IN_FLIGHT) return COLI_STRATA_PENDING;
        if (s->state == COLI_STRATA_SLOT_IN_USE) return COLI_STRATA_BUSY;
        if (s->state != COLI_STRATA_SLOT_READY) return COLI_STRATA_INVALID;
        if (p->backend.consume(p->backend.context, i, &s->identity, &event) != 1)
            return COLI_STRATA_BACKEND_ERROR;
        s->consumer_event = event; s->state = COLI_STRATA_SLOT_IN_USE;
        ticket->slot = i; ticket->generation = s->generation;
        return COLI_STRATA_OK;
    }
    return COLI_STRATA_MISS;
}

static inline coli_strata_status coli_strata_prefetch_release(coli_strata_prefetch *p,
        coli_strata_prefetch_ticket ticket) {
    coli_strata_prefetch_slot *s;
    int result, discard;
    if (!coli_strata_prefetch_valid_(p) || ticket.slot >= 2) return COLI_STRATA_INVALID;
    s = &p->slots[ticket.slot];
    if (s->state != COLI_STRATA_SLOT_IN_USE || !ticket.generation ||
        ticket.generation != s->generation) return COLI_STRATA_INVALID;
    result = p->backend.consumer_status(p->backend.context, s->consumer_event);
    if (result == 0) return COLI_STRATA_PENDING;
    if (result != 1 && result != -1) return COLI_STRATA_BACKEND_ERROR;
    discard = s->discard; coli_strata_prefetch_free_(s);
    if (result == -1) return COLI_STRATA_BACKEND_ERROR;
    return discard ? COLI_STRATA_CANCELLED : COLI_STRATA_OK;
}

static inline coli_strata_status coli_strata_prefetch_cancel(coli_strata_prefetch *p,
        coli_strata_prefetch_ticket ticket) {
    coli_strata_prefetch_slot *s;
    if (!coli_strata_prefetch_valid_(p) || ticket.slot >= 2) return COLI_STRATA_INVALID;
    s = &p->slots[ticket.slot];
    if (s->state == COLI_STRATA_SLOT_FREE || !ticket.generation ||
        ticket.generation != s->generation) return COLI_STRATA_INVALID;
    if (s->state == COLI_STRATA_SLOT_READY) {
        coli_strata_prefetch_free_(s); return COLI_STRATA_CANCELLED;
    }
    if (s->state != COLI_STRATA_SLOT_IN_FLIGHT && s->state != COLI_STRATA_SLOT_IN_USE)
        return COLI_STRATA_INVALID;
    s->discard = 1;
    return COLI_STRATA_PENDING;
}

#endif /* COLI_STRATA_MEMORY_H */

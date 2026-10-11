/* Caller-owned prepared activations for adjacent native CPU projections.
 * Inspired by UnbiasedStrata's shared ActQ / ExpertScratch contract at
 * d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf. This keeps the caller's quantizer:
 * it neither changes rounding nor memoizes mutable input pointers. */
#ifndef COLIBRI_STRATA_ACTIVATION_H
#define COLIBRI_STRATA_ACTIVATION_H
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#define STRATA_ACTIVATION_MAX_BYTES ((size_t)32 << 20)
#define STRATA_ACTIVATION_LOCAL_BYTES ((size_t)64 << 10)

typedef float (*StrataActivationQuantizer)(const float *, int, int8_t *, int32_t *);
typedef struct {
    const int8_t *q;
    const float *scale;
    int rows, columns;
} StrataActivation;

/* Subtraction avoids forming a wrapped end pointer. Nonempty wrapped ranges
 * are invalid; treating them as overlapping makes every caller reject them. */
static inline int strata_activation_overlap(const void *a, size_t an,
                                             const void *b, size_t bn) {
    uintptr_t ap = (uintptr_t)a, bp = (uintptr_t)b;
    if (!an || !bn) return 0;
    if (an > UINTPTR_MAX - ap || bn > UINTPTR_MAX - bp) return 1;
    return ap <= bp ? bp - ap < an : ap - bp < bn;
}

/* scales first, followed by signed INT8 rows. All limits are checked before
 * writing required, so failures preserve the caller's size sentinel. */
static inline int strata_activation_bytes(int rows, int columns, size_t limit,
                                           size_t *required) {
    if (!required || rows < 1 || columns < 1 || !limit ||
        limit > STRATA_ACTIVATION_MAX_BYTES) return 0;
    if ((size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)rows > SIZE_MAX / sizeof(float)) return 0;
    size_t values = (size_t)rows * (size_t)columns;
    size_t scales = (size_t)rows * sizeof(float);
    if (values > SIZE_MAX - scales || values + scales > limit) return 0;
    *required = values + scales;
    return 1;
}

/* x must contain finite activation values under quantize's existing contract.
 * No allocation; buffer must be float-aligned and separate from x/out. Every
 * validation failure leaves out and buffer untouched. Once prepared, readers
 * may share q/scale until the caller reuses or releases buffer. */
static inline int strata_activation_prepare(StrataActivation *out,
        const float *x, int rows, int columns, void *buffer, size_t capacity,
        StrataActivationQuantizer quantize) {
    size_t bytes;
    struct StrataFloatAlignment { char byte; float value; };
    size_t alignment = offsetof(struct StrataFloatAlignment, value);
    if (!out || !x || !buffer || !quantize ||
        !strata_activation_bytes(rows, columns, STRATA_ACTIVATION_MAX_BYTES, &bytes) ||
        capacity < bytes || (uintptr_t)buffer % alignment) return 0;
    size_t cells = (size_t)rows * (size_t)columns;
    if (cells > SIZE_MAX / sizeof(float)) return 0;
    size_t xbytes = cells * sizeof(float);
    if (strata_activation_overlap(buffer, bytes, x, xbytes) ||
        strata_activation_overlap(out, sizeof(*out), buffer, bytes) ||
        strata_activation_overlap(out, sizeof(*out), x, xbytes)) return 0;
    float *scale = (float *)buffer;
    int8_t *q = (int8_t *)((unsigned char *)buffer + (size_t)rows * sizeof(float));
    for (int row = 0; row < rows; row++)
        scale[row] = quantize(x + (size_t)row * columns, columns,
                             q + (size_t)row * columns, NULL);
    StrataActivation prepared = { q, scale, rows, columns };
    *out = prepared;
    return 1;
}
#endif

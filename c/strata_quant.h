/* Experimental, architecture-independent INT4/INT8/INT16 block weights.
 * Integer payloads are real signed quantized values; activations and outputs
 * are F32. matvec is a dequantize + double-accumulate CPU reference, not an
 * all-integer kernel, accelerator integration, or inference speed claim.
 * Quantization is LOSSY and requires model-specific quality evaluation.
 * All buffers are caller-owned; no allocations, device dependencies or I/O.
 */
#ifndef COLIBRI_STRATA_QUANT_H
#define COLIBRI_STRATA_QUANT_H

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "strata_quant requires finite checks: compile without fast-math/finite-math-only"
#endif

#include <limits.h>
#include "strata_memory.h" /* shared coli_strata_status and byte/range helpers */

#define STRATA_QUANT_MAX_VALUES UINT32_C(16777216)
#define STRATA_QUANT_MAX_BLOCK ((size_t)4096)
#define STRATA_QUANT_HEADER_BYTES ((size_t)24)
#define STRATA_QUANT_VERSION 1u

typedef struct {
    unsigned bits;
    size_t count, block_values, block_count, encoded_bytes;
} strata_quant_info;

/* CSQ1 wire format (little endian):
 *  0: magic[4], 4: u32 version=1, 8: u32 bits (4,8,16), 12: u32 count,
 * 16: u32 block_values, 20: u32 block_count, followed by blocks.
 * Each block stores an IEEE binary32 maxabs followed by its integer payload.
 * Scale=maxabs/qmax, computed in double to support subnormals and FLT_MAX.
 * INT4: signed two's complement [-7,7], first value in low nibble; an odd
 * block's unused high nibble is zero. INT8: [-127,127]. INT16: [-32767,32767]
 * in two little-endian bytes. The most negative representable code is invalid.
 * Canonical zero blocks store +0 maxabs and all-zero codes. Nonzero blocks
 * contain a +/-qmax code. Error <= maxabs/(2*qmax), plus F32 rounding.
 * Strict inspect/decode reject truncation, trailing bytes and noncanonical
 * blocks. Their blob is immutable. Results/output change only on success.
 * Array objects must actually have the supplied capacities; writable buffers
 * and result pointers must not alias each other or borrowed inputs.
 */

static inline int strata_quant_qmax_(unsigned bits) {
    return bits == 4 ? 7 : bits == 8 ? 127 : 32767;
}

static inline size_t strata_quant_payload_(size_t n, unsigned bits) {
    return bits == 4 ? n / 2 + (n % 2 != 0) : n * (bits / 8);
}

static inline coli_strata_status strata_quant_size(size_t count,
        size_t block_values, unsigned bits, size_t *bytes) {
    size_t full, tail, blocks, block_payload, payload, need;
    if (!bytes || !block_values || block_values > STRATA_QUANT_MAX_BLOCK ||
        (bits != 4 && bits != 8 && bits != 16)) return COLI_STRATA_INVALID;
    if (CHAR_BIT != 8 || sizeof(float) != 4 || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return COLI_STRATA_INVALID;
    if (count > STRATA_QUANT_MAX_VALUES) return COLI_STRATA_CAPACITY;
    full = count / block_values; tail = count % block_values;
    blocks = full + (tail != 0);
    block_payload = strata_quant_payload_(block_values, bits);
    if (full && block_payload > SIZE_MAX / full) return COLI_STRATA_OVERFLOW;
    payload = full * block_payload;
    if (strata_quant_payload_(tail, bits) > SIZE_MAX - payload) return COLI_STRATA_OVERFLOW;
    payload += strata_quant_payload_(tail, bits);
    if (blocks > (SIZE_MAX - STRATA_QUANT_HEADER_BYTES) / 4) return COLI_STRATA_OVERFLOW;
    need = STRATA_QUANT_HEADER_BYTES + blocks * 4;
    if (payload > SIZE_MAX - need) return COLI_STRATA_OVERFLOW;
    *bytes = need + payload;
    return COLI_STRATA_OK;
}

static inline int32_t strata_quant_read_q_(const uint8_t *payload,
        size_t index, unsigned bits) {
    uint32_t u;
    if (bits == 4) {
        u = (payload[index / 2] >> (index % 2 ? 4 : 0)) & 15u;
        return u > 7 ? (int32_t)u - 16 : (int32_t)u;
    }
    if (bits == 8) {
        u = payload[index];
        return u > 127 ? (int32_t)u - 256 : (int32_t)u;
    }
    u = (uint32_t)payload[index * 2] | ((uint32_t)payload[index * 2 + 1] << 8);
    return u > 32767 ? (int32_t)u - 65536 : (int32_t)u;
}

static inline void strata_quant_write_q_(uint8_t *payload,
        size_t index, unsigned bits, int32_t q) {
    uint32_t u = (uint32_t)(q < 0 ? q + (bits == 4 ? 16 : bits == 8 ? 256 : 65536) : q);
    if (bits == 4) {
        unsigned shift = index % 2 ? 4 : 0;
        payload[index / 2] = (uint8_t)(payload[index / 2] | (u << shift));
    } else if (bits == 8) payload[index] = (uint8_t)u;
    else {
        payload[index * 2] = (uint8_t)u;
        payload[index * 2 + 1] = (uint8_t)(u >> 8);
    }
}

static inline coli_strata_status strata_quant_encode(const float *src,
        size_t count, size_t block_values, unsigned bits, uint8_t *blob,
        size_t capacity, size_t *written) {
    size_t need, src_bytes, offset = STRATA_QUANT_HEADER_BYTES;
    int qmax;
    coli_strata_status status = strata_quant_size(count, block_values, bits, &need);
    if (status != COLI_STRATA_OK) return status;
    if (!blob || !written || (count && !src)) return COLI_STRATA_INVALID;
    if (capacity < need) return COLI_STRATA_CAPACITY;
    if (count > SIZE_MAX / sizeof(*src)) return COLI_STRATA_OVERFLOW;
    src_bytes = count * sizeof(*src);
    if (coli_strata_overlap_(src, src_bytes, blob, need) ||
        coli_strata_overlap_(written, sizeof(*written), blob, need) ||
        coli_strata_overlap_(written, sizeof(*written), src, src_bytes))
        return COLI_STRATA_INVALID;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(src[i])) return COLI_STRATA_NONFINITE;
    memcpy(blob, "CSQ1", 4);
    coli_strata_put_u32_(blob + 4, STRATA_QUANT_VERSION);
    coli_strata_put_u32_(blob + 8, bits);
    coli_strata_put_u32_(blob + 12, (uint32_t)count);
    coli_strata_put_u32_(blob + 16, (uint32_t)block_values);
    coli_strata_put_u32_(blob + 20,
            (uint32_t)(count / block_values + (count % block_values != 0)));
    qmax = strata_quant_qmax_(bits);
    for (size_t start = 0; start < count;) {
        size_t n = count - start, payload_bytes;
        float maxabs = 0;
        uint32_t maxbits;
        if (n > block_values) n = block_values;
        for (size_t i = 0; i < n; ++i) {
            float a = fabsf(src[start + i]);
            if (a > maxabs) maxabs = a;
        }
        memcpy(&maxbits, &maxabs, sizeof(maxbits));
        coli_strata_put_u32_(blob + offset, maxbits); offset += 4;
        payload_bytes = strata_quant_payload_(n, bits);
        memset(blob + offset, 0, payload_bytes);
        for (size_t i = 0; i < n; ++i) {
            double scaled = maxabs ? ((double)src[start + i] / maxabs) * qmax : 0;
            int32_t q = (int32_t)(scaled + (scaled < 0 ? -0.5 : 0.5));
            if (q > qmax) q = qmax;
            if (q < -qmax) q = -qmax;
            strata_quant_write_q_(blob + offset, i, bits, q);
        }
        offset += payload_bytes; start += n;
    }
    *written = need;
    return COLI_STRATA_OK;
}

static inline coli_strata_status strata_quant_inspect(const uint8_t *blob,
        size_t bytes, strata_quant_info *info) {
    strata_quant_info parsed = {0};
    size_t need, offset = STRATA_QUANT_HEADER_BYTES;
    uint32_t stored_bits, stored_count, stored_block, stored_blocks;
    int qmax;
    coli_strata_status status;
    if (!blob || !info || bytes < STRATA_QUANT_HEADER_BYTES) return COLI_STRATA_INVALID;
    if (memcmp(blob, "CSQ1", 4) || coli_strata_get_u32_(blob + 4) != STRATA_QUANT_VERSION)
        return COLI_STRATA_INVALID;
    stored_bits = coli_strata_get_u32_(blob + 8);
    stored_count = coli_strata_get_u32_(blob + 12);
    stored_block = coli_strata_get_u32_(blob + 16);
    stored_blocks = coli_strata_get_u32_(blob + 20);
    if (stored_bits != 4 && stored_bits != 8 && stored_bits != 16) return COLI_STRATA_INVALID;
#if SIZE_MAX < UINT32_MAX
    if (stored_count > SIZE_MAX || stored_block > SIZE_MAX || stored_blocks > SIZE_MAX)
        return COLI_STRATA_CAPACITY;
#endif
    parsed.bits = (unsigned)stored_bits; parsed.count = (size_t)stored_count;
    parsed.block_values = (size_t)stored_block; parsed.block_count = (size_t)stored_blocks;
    status = strata_quant_size(parsed.count, parsed.block_values, parsed.bits, &need);
    if (status != COLI_STRATA_OK) return status;
    if (bytes != need || parsed.block_count != parsed.count / parsed.block_values +
            (parsed.count % parsed.block_values != 0)) return COLI_STRATA_INVALID;
    if (coli_strata_overlap_(info, sizeof(*info), blob, bytes)) return COLI_STRATA_INVALID;
    qmax = strata_quant_qmax_(parsed.bits);
    for (size_t start = 0; start < parsed.count;) {
        size_t n = parsed.count - start;
        uint32_t maxbits = coli_strata_get_u32_(blob + offset);
        float maxabs;
        int has_extremum = 0;
        if (n > parsed.block_values) n = parsed.block_values;
        memcpy(&maxabs, &maxbits, sizeof(maxabs));
        /* Reject negative zero as well as negative/nonfinite scales. */
        if (!isfinite(maxabs) || (maxbits >> 31)) return COLI_STRATA_INVALID;
        offset += 4;
        for (size_t i = 0; i < n; ++i) {
            int32_t q = strata_quant_read_q_(blob + offset, i, parsed.bits);
            if (q < -qmax || q > qmax || (!maxabs && q)) return COLI_STRATA_INVALID;
            if (q == qmax || q == -qmax) has_extremum = 1;
        }
        if (maxabs && !has_extremum) return COLI_STRATA_INVALID;
        if (parsed.bits == 4 && n % 2 && (blob[offset + n / 2] & 0xf0))
            return COLI_STRATA_INVALID;
        offset += strata_quant_payload_(n, parsed.bits); start += n;
    }
    parsed.encoded_bytes = need; *info = parsed;
    return COLI_STRATA_OK;
}

static inline coli_strata_status strata_quant_decode(const uint8_t *blob,
        size_t bytes, float *dst, size_t capacity, size_t *decoded) {
    strata_quant_info info;
    size_t offset = STRATA_QUANT_HEADER_BYTES;
    int qmax;
    coli_strata_status status;
    if (!decoded) return COLI_STRATA_INVALID;
    status = strata_quant_inspect(blob, bytes, &info);
    if (status != COLI_STRATA_OK) return status;
    if (capacity < info.count) return COLI_STRATA_CAPACITY;
    if (info.count && !dst) return COLI_STRATA_INVALID;
    if (info.count > SIZE_MAX / sizeof(*dst)) return COLI_STRATA_OVERFLOW;
    if (coli_strata_overlap_(blob, bytes, dst, info.count * sizeof(*dst)) ||
        coli_strata_overlap_(decoded, sizeof(*decoded), blob, bytes) ||
        coli_strata_overlap_(decoded, sizeof(*decoded), dst, info.count * sizeof(*dst)))
        return COLI_STRATA_INVALID;
    qmax = strata_quant_qmax_(info.bits);
    for (size_t start = 0; start < info.count;) {
        size_t n = info.count - start;
        uint32_t maxbits = coli_strata_get_u32_(blob + offset);
        float maxabs;
        memcpy(&maxabs, &maxbits, sizeof(maxabs));
        if (n > info.block_values) n = info.block_values;
        offset += 4;
        for (size_t i = 0; i < n; ++i)
            dst[start + i] = (float)(strata_quant_read_q_(blob + offset, i, info.bits) *
                                    ((double)maxabs / qmax));
        offset += strata_quant_payload_(n, info.bits); start += n;
    }
    *decoded = info.count;
    return COLI_STRATA_OK;
}

/* Row-major matrix; block boundaries can cross rows. x_count must equal cols,
 * y_count >= rows, workspace_count >= rows*cols. Workspace first holds the
 * decoded F32 weights; after each row is read, its result is staged at the
 * beginning of workspace (which cannot overwrite an unread future row).
 * Only after every row passes arithmetic checks are results copied to y.
 * Workspace contents may change on error, y/inputs/blob never do. Float
 * output overflow returns OVERFLOW; nonfinite activations return NONFINITE.
 * This simple reference requires matrix-sized scratch and uses no accelerator.
 */
static inline coli_strata_status strata_quant_matvec(const uint8_t *blob,
        size_t bytes, size_t rows, size_t cols, const float *x, size_t x_count,
        float *y, size_t y_count, float *workspace, size_t workspace_count) {
    strata_quant_info info;
    size_t count, decoded;
    coli_strata_status status;
    if (!rows || !cols || !x || !y || !workspace) return COLI_STRATA_INVALID;
    if (rows > SIZE_MAX / cols) return COLI_STRATA_OVERFLOW;
    count = rows * cols;
    if (count > STRATA_QUANT_MAX_VALUES) return COLI_STRATA_CAPACITY;
    if (x_count != cols) return COLI_STRATA_INVALID;
    if (y_count < rows || workspace_count < count) return COLI_STRATA_CAPACITY;
    if (x_count > SIZE_MAX / sizeof(*x) || y_count > SIZE_MAX / sizeof(*y) ||
        workspace_count > SIZE_MAX / sizeof(*workspace)) return COLI_STRATA_OVERFLOW;
    status = strata_quant_inspect(blob, bytes, &info);
    if (status != COLI_STRATA_OK) return status;
    if (info.count != count) return COLI_STRATA_INVALID;
    {
        const void *buffers[] = {blob, x, y, workspace};
        size_t sizes[] = {bytes, x_count * sizeof(*x), y_count * sizeof(*y),
                         workspace_count * sizeof(*workspace)};
        for (size_t i = 1; i < 4; ++i)
            for (size_t j = 0; j < i; ++j)
                if (coli_strata_overlap_(buffers[i], sizes[i], buffers[j], sizes[j]))
                    return COLI_STRATA_INVALID;
    }
    for (size_t i = 0; i < cols; ++i)
        if (!isfinite(x[i])) return COLI_STRATA_NONFINITE;
    status = strata_quant_decode(blob, bytes, workspace, workspace_count, &decoded);
    if (status != COLI_STRATA_OK) return status;
    for (size_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (size_t col = 0; col < cols; ++col)
            sum += (double)workspace[row * cols + col] * x[col];
        if (!isfinite(sum)) return COLI_STRATA_NONFINITE;
        if (sum > FLT_MAX || sum < -FLT_MAX) return COLI_STRATA_OVERFLOW;
        workspace[row] = (float)sum;
    }
    memcpy(y, workspace, rows * sizeof(*y));
    return COLI_STRATA_OK;
}

#endif /* COLIBRI_STRATA_QUANT_H */

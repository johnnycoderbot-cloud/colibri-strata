/* Quantized payload, transactional failure, and double-reference behavior.
 * No model weights or accelerators are needed. */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../strata_quant.h"

static uint32_t rng = 0x9482fa35u;
static uint32_t random_u32(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng;
}

static void expect_bad(const uint8_t *blob, size_t bytes) {
    strata_quant_info info, original_info;
    float dst[300], original[300];
    size_t decoded = 0xf00d;
    memset(&info, 0xa5, sizeof(info)); memcpy(&original_info, &info, sizeof(info));
    for (size_t i = 0; i < 300; ++i) dst[i] = (float)i + 0.25f;
    memcpy(original, dst, sizeof(dst));
    assert(strata_quant_inspect(blob, bytes, &info) != COLI_STRATA_OK);
    assert(memcmp(&info, &original_info, sizeof(info)) == 0);
    assert(strata_quant_decode(blob, bytes, dst, 300, &decoded) != COLI_STRATA_OK);
    assert(decoded == 0xf00d && memcmp(original, dst, sizeof(dst)) == 0);
}

static void test_wire(void) {
    const unsigned widths[] = {4, 8, 16};
    const uint8_t four[] = {0xf9, 0x10, 0x07};
    const uint8_t eight[] = {0x81, 0xff, 0x00, 0x01, 0x7f};
    const uint8_t sixteen[] = {0x01, 0x80, 0xff, 0xff, 0, 0, 1, 0, 0xff, 0x7f};
    for (size_t k = 0; k < 3; ++k) {
        unsigned bits = widths[k];
        int qmax = bits == 4 ? 7 : bits == 8 ? 127 : 32767;
        float src[] = {-(float)qmax, -1, 0, 1, (float)qmax}, dst[5];
        uint8_t blob[128], snapshot[128];
        size_t written, decoded;
        strata_quant_info info;
        assert(strata_quant_encode(src, 5, 5, bits, blob, sizeof(blob), &written) == COLI_STRATA_OK);
        assert(!memcmp(blob, "CSQ1", 4));
        assert(coli_strata_get_u32_(blob + 4) == 1 && coli_strata_get_u32_(blob + 8) == bits);
        assert(coli_strata_get_u32_(blob + 12) == 5 && coli_strata_get_u32_(blob + 16) == 5);
        assert(coli_strata_get_u32_(blob + 20) == 1);
        if (bits == 4) assert(written == 31 && !memcmp(blob + 28, four, sizeof(four)));
        if (bits == 8) assert(written == 33 && !memcmp(blob + 28, eight, sizeof(eight)));
        if (bits == 16) assert(written == 38 && !memcmp(blob + 28, sixteen, sizeof(sixteen)));
        assert(strata_quant_inspect(blob, written, &info) == COLI_STRATA_OK);
        assert(info.bits == bits && info.count == 5 && info.block_values == 5 &&
               info.block_count == 1 && info.encoded_bytes == written);
        memcpy(snapshot, blob, written);
        assert(strata_quant_decode(blob, written, dst, 5, &decoded) == COLI_STRATA_OK && decoded == 5);
        assert(!memcmp(snapshot, blob, written));
        for (size_t i = 0; i < 5; ++i) assert(dst[i] == src[i]);
    }
    {
        float ties[] = {0.5f, -0.5f, 7};
        uint8_t blob[64]; size_t written;
        assert(strata_quant_encode(ties, 3, 3, 4, blob, sizeof(blob), &written) == COLI_STRATA_OK);
        assert(blob[28] == 0xf1 && blob[29] == 0x07); /* nearest, ties away from zero */
    }
}

static void test_properties(void) {
    const unsigned widths[] = {4, 8, 16};
    const size_t blocks[] = {1, 2, 3, 7, 15, 32, 63, 257, 4096};
    for (size_t width = 0; width < 3; ++width) {
        unsigned bits = widths[width];
        int qmax = bits == 4 ? 7 : bits == 8 ? 127 : 32767;
        for (size_t iteration = 0; iteration < 180; ++iteration) {
            float src[257], dst[257];
            uint8_t blob[2048], snapshot[2048], reencoded[2048];
            size_t n = iteration < 10 ? iteration : random_u32() % 258;
            size_t block = blocks[iteration % (sizeof(blocks) / sizeof(blocks[0]))];
            size_t need, written, decoded, rewritten;
            strata_quant_info info;
            for (size_t i = 0; i < n; ++i) {
                int v = (int)(random_u32() % 20001) - 10000;
                src[i] = (float)(v * 0.001 * pow(10.0, (int)(i % 7) - 3));
            }
            assert(strata_quant_size(n, block, bits, &need) == COLI_STRATA_OK && need <= sizeof(blob));
            assert(strata_quant_encode(src, n, block, bits, blob, sizeof(blob), &written)
                   == COLI_STRATA_OK && written == need);
            memcpy(snapshot, blob, written);
            assert(strata_quant_inspect(blob, written, &info) == COLI_STRATA_OK);
            assert(info.bits == bits && info.count == n && info.block_values == block &&
                   info.block_count == n / block + (n % block != 0) && info.encoded_bytes == written);
            assert(strata_quant_decode(blob, written, dst, n, &decoded) == COLI_STRATA_OK && decoded == n);
            assert(!memcmp(snapshot, blob, written));
            for (size_t start = 0; start < n; start += block) {
                size_t end = start + block < n ? start + block : n;
                double maxabs = 0;
                for (size_t i = start; i < end; ++i)
                    if (fabs(src[i]) > maxabs) maxabs = fabs(src[i]);
                for (size_t i = start; i < end; ++i) {
                    assert(isfinite(dst[i]));
                    assert(fabs((double)src[i] - dst[i]) <= maxabs / (2 * qmax) + maxabs * FLT_EPSILON);
                }
            }
            assert(strata_quant_encode(dst, n, block, bits, reencoded, sizeof(reencoded), &rewritten)
                   == COLI_STRATA_OK && rewritten == written);
            assert(!memcmp(blob, reencoded, written));
        }
        {
            float src[] = {FLT_MAX, -FLT_MAX, 0, -0.0f,
                           nextafterf(0, 1), -nextafterf(0, 1), 0};
            float dst[7]; uint8_t blob[128]; size_t written, decoded;
            assert(strata_quant_encode(src, 7, 2, bits, blob, sizeof(blob), &written) == COLI_STRATA_OK);
            assert(strata_quant_decode(blob, written, dst, 7, &decoded) == COLI_STRATA_OK);
            for (size_t i = 0; i < 7; ++i) assert(isfinite(dst[i]) && src[i] == dst[i]);
            assert(strata_quant_encode(NULL, 0, 2, bits, blob, sizeof(blob), &written)
                   == COLI_STRATA_OK && written == 24);
            assert(strata_quant_decode(blob, written, NULL, 0, &decoded)
                   == COLI_STRATA_OK && decoded == 0);
        }
    }
    {
        size_t four, eight, sixteen;
        assert(strata_quant_size(257, 32, 4, &four) == COLI_STRATA_OK && four == 189);
        assert(strata_quant_size(257, 32, 8, &eight) == COLI_STRATA_OK && eight == 317);
        assert(strata_quant_size(257, 32, 16, &sixteen) == COLI_STRATA_OK && sixteen == 574);
        assert(four < eight && eight < sixteen && sixteen < 257 * sizeof(float));
        assert(strata_quant_size(6, 3, 4, &four) == COLI_STRATA_OK && four == 36);
    }
}

static void test_invalid_and_alias(void) {
    const unsigned widths[] = {4, 8, 16};
    size_t size = 123;
    assert(strata_quant_size(1, 0, 4, &size) == COLI_STRATA_INVALID && size == 123);
    assert(strata_quant_size(1, STRATA_QUANT_MAX_BLOCK + 1, 4, &size) == COLI_STRATA_INVALID);
    assert(strata_quant_size(1, 32, 3, &size) == COLI_STRATA_INVALID);
    assert(strata_quant_size(1, 32, 32, &size) == COLI_STRATA_INVALID);
    assert(strata_quant_size(1, 32, 8, NULL) == COLI_STRATA_INVALID);
    assert(strata_quant_size(STRATA_QUANT_MAX_VALUES + 1, 32, 16, &size) == COLI_STRATA_CAPACITY);
    assert(strata_quant_size(SIZE_MAX, 32, 16, &size) == COLI_STRATA_CAPACITY);
    assert(strata_quant_size(STRATA_QUANT_MAX_VALUES, STRATA_QUANT_MAX_BLOCK, 16, &size)
           == COLI_STRATA_OK && size == 24 + 4 * 4096 + 2 * STRATA_QUANT_MAX_VALUES);
    for (size_t width = 0; width < 3; ++width) {
        unsigned bits = widths[width];
        float src[] = {7, -7, 0}, dst[3] = {3, 4, 5};
        uint8_t blob[128], mutated[128], original[128];
        size_t written, decoded = 123;
        const size_t header_offsets[] = {4, 8, 12, 16, 20};
        const uint32_t bad_values[] = {2, 7, UINT32_MAX, 0, 0};
        assert(strata_quant_encode(src, 3, 3, bits, blob, sizeof(blob), &written) == COLI_STRATA_OK);
        for (size_t truncated = 0; truncated < written; ++truncated) expect_bad(blob, truncated);
        expect_bad(blob, written + 1);
        memcpy(mutated, blob, written); mutated[0] = 0; expect_bad(mutated, written);
        for (size_t i = 0; i < 5; ++i) {
            memcpy(mutated, blob, written);
            coli_strata_put_u32_(mutated + header_offsets[i], bad_values[i]);
            expect_bad(mutated, written);
        }
        {
            const uint32_t bad_scales[] = {0x80000000u, 0xbf800000u, 0x7f800000u, 0x7fc00000u};
            for (size_t i = 0; i < 4; ++i) {
                memcpy(mutated, blob, written);
                coli_strata_put_u32_(mutated + 24, bad_scales[i]);
                expect_bad(mutated, written);
            }
        }
        memcpy(mutated, blob, written);
        if (bits == 4) mutated[28] = (uint8_t)((mutated[28] & 0xf0) | 8);
        if (bits == 8) mutated[28] = 128;
        if (bits == 16) { mutated[28] = 0; mutated[29] = 128; }
        expect_bad(mutated, written);
        memcpy(mutated, blob, written); memset(mutated + 28, 0, written - 28);
        expect_bad(mutated, written); /* nonzero scale with no extremum */
        memcpy(mutated, blob, written); coli_strata_put_u32_(mutated + 24, 0);
        expect_bad(mutated, written); /* nonzero values in zero block */
        if (bits == 4) {
            memcpy(mutated, blob, written); mutated[29] |= 0x10;
            expect_bad(mutated, written); /* unused odd-block nibble */
        }
        assert(strata_quant_decode(blob, written, dst, 2, &decoded) == COLI_STRATA_CAPACITY);
        assert(decoded == 123 && dst[0] == 3 && dst[1] == 4 && dst[2] == 5);
        assert(strata_quant_decode(blob, written, NULL, 3, &decoded) == COLI_STRATA_INVALID);
        assert(strata_quant_decode(blob, written, dst, 3, NULL) == COLI_STRATA_INVALID);
        assert(strata_quant_inspect(blob, written, NULL) == COLI_STRATA_INVALID);

        memset(blob, 0xa5, sizeof(blob)); memcpy(original, blob, sizeof(blob)); written = 123;
        src[2] = NAN;
        assert(strata_quant_encode(src, 3, 3, bits, blob, sizeof(blob), &written) == COLI_STRATA_NONFINITE);
        assert(!memcmp(blob, original, sizeof(blob)) && written == 123);
        src[2] = -INFINITY;
        assert(strata_quant_encode(src, 3, 3, bits, blob, sizeof(blob), &written) == COLI_STRATA_NONFINITE);
        src[2] = 0;
        assert(strata_quant_encode(src, 3, 3, bits, blob, 1, &written) == COLI_STRATA_CAPACITY);
        assert(strata_quant_encode(NULL, 3, 3, bits, blob, sizeof(blob), &written) == COLI_STRATA_INVALID);
        assert(strata_quant_encode(src, 3, 3, bits, NULL, sizeof(blob), &written) == COLI_STRATA_INVALID);
        assert(strata_quant_encode(src, 3, 3, bits, blob, sizeof(blob), NULL) == COLI_STRATA_INVALID);
        assert(!memcmp(blob, original, sizeof(blob)) && written == 123);
        {
            union { uint8_t b[256]; float f[64]; strata_quant_info info; size_t n; } alias;
            alias.f[0] = 1;
            assert(strata_quant_encode(alias.f, 1, 1, bits, alias.b, sizeof(alias), &written)
                   == COLI_STRATA_INVALID);
            assert(strata_quant_encode(src, 3, 3, bits, alias.b, sizeof(alias), &alias.n)
                   == COLI_STRATA_INVALID);
            assert(strata_quant_encode(src, 3, 3, bits, alias.b, sizeof(alias), &written)
                   == COLI_STRATA_OK);
            assert(strata_quant_inspect(alias.b, written, &alias.info) == COLI_STRATA_INVALID);
            assert(strata_quant_decode(alias.b, written, alias.f, 64, &decoded) == COLI_STRATA_INVALID);
            assert(strata_quant_decode(alias.b, written, dst, 3, &alias.n) == COLI_STRATA_INVALID);
        }
    }
}

static void test_matvec(void) {
    const unsigned widths[] = {4, 8, 16};
    const size_t shapes[][2] = {{4, 5}, {5, 1}, {1, 31}, {7, 7}};
    for (size_t width = 0; width < 3; ++width) {
        for (size_t shape = 0; shape < 4; ++shape) {
            size_t rows = shapes[shape][0], cols = shapes[shape][1], n = rows * cols;
            float weights[64], decoded[64], workspace[64], x[64], y[64];
            uint8_t blob[512], snapshot[512]; size_t written, count;
            for (size_t i = 0; i < n; ++i) weights[i] = (float)((int)(random_u32() % 1001) - 500) / 103;
            for (size_t i = 0; i < cols; ++i) x[i] = (float)((int)(random_u32() % 501) - 250) / 97;
            for (size_t i = 0; i < 64; ++i) y[i] = 1234;
            assert(strata_quant_encode(weights, n, 3, widths[width], blob, sizeof(blob), &written)
                   == COLI_STRATA_OK);
            memcpy(snapshot, blob, written);
            assert(strata_quant_decode(blob, written, decoded, 64, &count) == COLI_STRATA_OK);
            assert(strata_quant_matvec(blob, written, rows, cols, x, cols, y, 64, workspace, 64)
                   == COLI_STRATA_OK);
            for (size_t row = 0; row < rows; ++row) {
                double expected = 0;
                for (size_t col = 0; col < cols; ++col) expected += (double)decoded[row * cols + col] * x[col];
                assert(y[row] == (float)expected && isfinite(y[row]));
            }
            for (size_t i = rows; i < 64; ++i) assert(y[i] == 1234);
            assert(!memcmp(snapshot, blob, written));
        }
        {
            float weights[] = {1, 1, FLT_MAX, FLT_MAX};
            float x[4] = {1, 1, 0, 0}, y[] = {123, 456}, workspace[4];
            uint8_t blob[128]; size_t written;
            assert(strata_quant_encode(weights, 4, 2, widths[width], blob, sizeof(blob), &written)
                   == COLI_STRATA_OK);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, 2, workspace, 4)
                   == COLI_STRATA_OVERFLOW && y[0] == 123 && y[1] == 456);
            x[1] = NAN;
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, 2, workspace, 4)
                   == COLI_STRATA_NONFINITE && y[0] == 123 && y[1] == 456);
            x[1] = 1;
            assert(strata_quant_matvec(blob, written, 0, 2, x, 2, y, 2, workspace, 4) == COLI_STRATA_INVALID);
            assert(strata_quant_matvec(blob, written, SIZE_MAX, 2, x, 2, y, 2, workspace, 4) == COLI_STRATA_OVERFLOW);
            assert(strata_quant_matvec(blob, written, STRATA_QUANT_MAX_VALUES + 1, 1,
                   x, 1, y, 2, workspace, 4) == COLI_STRATA_CAPACITY);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 1, y, 2, workspace, 4) == COLI_STRATA_INVALID);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, 1, workspace, 4) == COLI_STRATA_CAPACITY);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, 2, workspace, 3) == COLI_STRATA_CAPACITY);
            assert(strata_quant_matvec(blob, written, 2, 2, NULL, 2, y, 2, workspace, 4) == COLI_STRATA_INVALID);
            assert(strata_quant_matvec(blob, written, 1, 2, x, 2, y, 2, workspace, 4) == COLI_STRATA_INVALID);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, x, 2, workspace, 4) == COLI_STRATA_INVALID);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, 2, x, 4) == COLI_STRATA_INVALID);
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, SIZE_MAX, workspace, 4) == COLI_STRATA_OVERFLOW);
            assert(y[0] == 123 && y[1] == 456 && x[0] == 1 && x[1] == 1);
            blob[0] = 0;
            assert(strata_quant_matvec(blob, written, 2, 2, x, 2, y, 2, workspace, 4) == COLI_STRATA_INVALID);
            assert(y[0] == 123 && y[1] == 456);
        }
    }
}

int main(void) {
    test_wire();
    test_properties();
    test_invalid_and_alias();
    test_matvec();
    puts("test_strata_quant: ok");
    return 0;
}

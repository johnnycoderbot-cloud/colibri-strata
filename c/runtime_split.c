#include "runtime_split.h"
#include <float.h>
#include <stddef.h>
#include <string.h>

/* The ABI uses IEEE binary64 microseconds. Integer arithmetic keeps validation,
 * overflow detection, subnormal costs and lower-q ties intact under fast-math
 * (including a process that enables flush-to-zero for floating-point math). */
#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "runtime split requires IEEE binary64 double"
#endif
_Static_assert(sizeof(double) == sizeof(uint64_t), "binary64 ABI size");
#define SPLIT_FRAC UINT64_C(0x000fffffffffffff)
#define SPLIT_MAG  UINT64_C(0x7fffffffffffffff)
#define SPLIT_EXP  UINT64_C(0x7ff0000000000000)
#define SPLIT_HIDDEN UINT64_C(0x0010000000000000)

static int split_cost(const double *input, uint64_t *bits) {
    uint64_t value;
    memcpy(&value, input, sizeof value);
    if ((value & SPLIT_EXP) == SPLIT_EXP ||
        ((value >> 63) && (value & SPLIT_MAG))) return 0;
    *bits = value & SPLIT_MAG; /* Both zero signs mean unavailable cold work. */
    return 1;
}

static uint64_t split_shift_jam(uint64_t value, unsigned shift) {
    if (!shift) return value;
    if (shift >= 64) return value != 0;
    return (value >> shift) | ((value & ((UINT64_C(1) << shift) - 1)) != 0);
}

/* Round positive addition to nearest, ties to even, exactly as binary64. */
static int split_add(uint64_t a, uint64_t b, uint64_t *result) {
    if (!a) { *result = b; return 1; }
    if (!b) { *result = a; return 1; }
    if (a < b) { uint64_t temp = a; a = b; b = temp; }
    unsigned ea = (unsigned)(a >> 52), eb = (unsigned)(b >> 52);
    uint64_t ma = a & SPLIT_FRAC, mb = b & SPLIT_FRAC;
    if (ea) ma |= SPLIT_HIDDEN; else ea = 1;
    if (eb) mb |= SPLIT_HIDDEN; else eb = 1;
    uint64_t sum = (ma << 3) + split_shift_jam(mb << 3, ea - eb);
    if (sum >= (SPLIT_HIDDEN << 4)) {
        sum = split_shift_jam(sum, 1);
        ++ea;
    }
    uint64_t significand = sum >> 3, tail = sum & 7;
    if (tail > 4 || (tail == 4 && (significand & 1))) ++significand;
    if (significand >= (SPLIT_HIDDEN << 1)) {
        significand >>= 1;
        ++ea;
    }
    if (ea >= 2047) return 0;
    *result = (ea == 1 && significand < SPLIT_HIDDEN) ? significand :
              ((uint64_t)ea << 52) | (significand & SPLIT_FRAC);
    return 1;
}

static unsigned split_high_bit(uint64_t value) {
    unsigned bit = 0;
    if (value >> 32) { value >>= 32; bit += 32; }
    if (value >> 16) { value >>= 16; bit += 16; }
    if (value >> 8) { value >>= 8; bit += 8; }
    if (value >> 4) { value >>= 4; bit += 4; }
    if (value >> 2) { value >>= 2; bit += 2; }
    if (value >> 1) bit += 1;
    return bit;
}

/* n <= 1024: the exact significand product fits in uint64_t. */
static int split_multiply(uint64_t value, uint32_t n, uint64_t *result) {
    if (!value || !n) { *result = 0; return 1; }
    unsigned exponent = (unsigned)(value >> 52);
    uint64_t significand = value & SPLIT_FRAC;
    if (exponent) significand |= SPLIT_HIDDEN; else exponent = 1;
    uint64_t product = significand * (uint64_t)n;
    unsigned high = split_high_bit(product);
    if (high > 52) {
        unsigned shift = high - 52;
        uint64_t tail = product & ((UINT64_C(1) << shift) - 1);
        uint64_t half = UINT64_C(1) << (shift - 1);
        significand = product >> shift;
        if (tail > half || (tail == half && (significand & 1))) ++significand;
        exponent += shift;
    } else significand = product;
    if (significand >= (SPLIT_HIDDEN << 1)) {
        significand >>= 1;
        ++exponent;
    }
    if (exponent >= 2047) return 0;
    *result = (exponent == 1 && significand < SPLIT_HIDDEN) ? significand :
              ((uint64_t)exponent << 52) | (significand & SPLIT_FRAC);
    return 1;
}

int coli_split_plan(const uint32_t controls[4], const double costs[9],
                    uint32_t counts[2], double times[4]) {
    uint32_t control[4]; uint64_t cost[9], chosen[4] = {0,0,0,0};
    uint32_t device_count = 0; int found = 0;
    if (!controls || !costs || !counts || !times) return 1;
    /* Output spans must be distinct; inputs are snapshotted before any write. */
    uintptr_t cp = (uintptr_t)counts, tp = (uintptr_t)times;
    if (cp > UINTPTR_MAX - 2 * sizeof *counts || tp > UINTPTR_MAX - 4 * sizeof *times ||
        (cp < tp + 4 * sizeof *times && tp < cp + 2 * sizeof *counts)) return 1;
    memcpy(control, controls, sizeof control);
    if (control[0] != COLI_SPLIT_ABI || control[1] > COLI_SPLIT_MAX_EXPERTS ||
        control[2] > control[1] || control[3] > 1) return 1;
    for (unsigned i = 0; i < 9; ++i) if (!split_cost(costs + i, cost + i)) return 1;
    for (uint32_t q = 0; q <= control[2]; ++q) {
        uint32_t h = control[1] - q;
        uint64_t host, device, transfer, serial = cost[8], total;
        if ((h && !cost[0]) || (q && !cost[2])) continue;
        if (!split_multiply(cost[0], h, &host) || !split_add(host, cost[3], &host) ||
            ((h || cost[3]) && !split_add(host, cost[5], &host)) ||
            !split_multiply(cost[2], q, &device) || !split_add(device, cost[4], &device) ||
            ((q || cost[4]) && !split_add(device, cost[6], &device)) ||
            !split_multiply(cost[1], q, &transfer)) continue;
        if (control[3]) {
            if (!split_add(device, transfer, &device)) continue;
        } else if (!split_add(serial, transfer, &serial)) continue;
        if (!split_add(serial, host > device ? host : device, &total) ||
            !split_add(total, cost[7], &total)) continue;
        if (!found || total < chosen[3]) {
            chosen[0] = host; chosen[1] = device; chosen[2] = serial; chosen[3] = total;
            device_count = q; found = 1;
        }
    }
    if (!found) return 2;
    uint32_t selected[2] = {device_count, control[1] - device_count};
    memcpy(counts, selected, sizeof selected);
    memcpy(times, chosen, sizeof chosen);
    return 0;
}

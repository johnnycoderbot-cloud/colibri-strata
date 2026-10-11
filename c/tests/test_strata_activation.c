/* Native shared-activation and projection-group numerical/ownership gates.
 * No model download, training, accelerator, or throughput measurement. */
#define COLI_QWEN_ACTIVATION_TEST 1
#ifdef COLI_CUDA
#undef COLI_CUDA
#endif
#ifdef COLI_VULKAN
#undef COLI_VULKAN
#endif
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include <assert.h>

static int integer_mode, group_mode;
static uint32_t seed = 0x39f0e27u;
static uint32_t next_random(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed;
}
static float sample_value(size_t i) {
    return (float)((int)((i * 37 + 23) % 257) - 128) / 97.f;
}
static void counters_reset(void) {
    g_qwen_activation_quant_calls = g_qwen_projection_group_dispatches = 0;
    g_qwen_activation_heap_allocs = 0;
}
static QW weights(int I, int O) {
    QW w; memset(&w, 0, sizeof w); w.I = I; w.O = O;
    w.q = malloc((size_t)I * O); w.sc = falloc(O); assert(w.q);
    for (size_t i = 0; i < (size_t)I * O; i++) w.q[i] = (int8_t)((int)(next_random() % 256) - 128);
    for (int o = 0; o < O; o++) w.sc[o] = (float)(o % 11 + 1) / 128.f;
    return w;
}
static void weight_free(QW *w) {
    free(w->q); free(w->sc); free((void *)w->w); free(w->h); free(w->q4); free(w->sg);
}

static void prepared_activation_gates(void) {
    union { float alignment; unsigned char bytes[4096]; } storage;
    float x[5 * 129]; int8_t q[5 * 129]; float scales[5];
    for (size_t i = 0; i < sizeof x / sizeof x[0]; i++) x[i] = sample_value(i);
    StrataActivation got;
    size_t required = 99;
    assert(strata_activation_bytes(5, 129, sizeof storage.bytes, &required));
    assert(required == 5 * (129 + sizeof(float)));
    memset(storage.bytes, 0xa7, sizeof storage.bytes);
    assert(strata_activation_prepare(&got, x, 5, 129, storage.bytes, required, dense_act_i8));
    for (int s = 0; s < 5; s++) scales[s] = dense_act_i8(x + s * 129, 129, q + s * 129, NULL);
    assert(got.rows == 5 && got.columns == 129);
    assert(!memcmp(got.q, q, sizeof q)); assert(!memcmp(got.scale, scales, sizeof scales));
    for (size_t i = required; i < sizeof storage.bytes; i++) assert(storage.bytes[i] == 0xa7);

    StrataActivation sentinel; memset(&sentinel, 0x33, sizeof sentinel);
    const int rows[] = {0, -1, INT_MAX, 1, 1};
    const int cols[] = {129, 129, INT_MAX, 0, -1};
    for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++) {
        size_t unchanged = 121;
        assert(!strata_activation_bytes(rows[i], cols[i], STRATA_ACTIVATION_MAX_BYTES, &unchanged));
        assert(unchanged == 121);
    }
    assert(!strata_activation_bytes(1, 1, 0, &required));
    assert(!strata_activation_bytes(1, 1, STRATA_ACTIVATION_MAX_BYTES + 1, &required));
    assert(!strata_activation_bytes(1, 1, 8, NULL));
    got = sentinel; memset(storage.bytes, 0x67, sizeof storage.bytes);
    assert(!strata_activation_prepare(&got, x, 5, 129, storage.bytes, 10, dense_act_i8));
    assert(!memcmp(&got, &sentinel, sizeof got));
    for (size_t i = 0; i < sizeof storage.bytes; i++) assert(storage.bytes[i] == 0x67);
    assert(!strata_activation_prepare(&got, x, 5, 129, storage.bytes + 1, 1000, dense_act_i8));
    assert(!strata_activation_prepare(&got, x, 5, 129, x, sizeof x, dense_act_i8));
    assert(!strata_activation_prepare((StrataActivation *)storage.bytes, x, 5, 129,
                                      storage.bytes, sizeof storage.bytes, dense_act_i8));
    assert(!strata_activation_prepare(&got, x, 5, 129, storage.bytes, sizeof storage.bytes, NULL));
    assert(!strata_activation_prepare(&got, NULL, 5, 129, storage.bytes, sizeof storage.bytes, dense_act_i8));
    assert(!memcmp(&got, &sentinel, sizeof got));
    assert(strata_activation_overlap((void *)(uintptr_t)(UINTPTR_MAX - 3), 8, x, sizeof x));
    assert(!strata_activation_overlap(x, 0, x, sizeof x));
    memset(x, 0, sizeof x);
    assert(strata_activation_prepare(&got, x, 5, 129, storage.bytes, sizeof storage.bytes, dense_act_i8));
    for (int s = 0; s < 5; s++) assert(got.scale[s] == 1e-12f);
    for (int i = 0; i < 5 * 129; i++) assert(got.q[i] == 0);
}

static void one_group(int S, int I, int count, int format, int heap_fail, int cap_fail) {
    QW w[3]; Q36Projection p[3]; float *want[3], *owned[3];
    const int outputs[] = {3, 17, 33};
    float *x = falloc((int64_t)S * I);
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = sample_value(i);
    for (int j = 0; j < count; j++) {
        w[j] = weights(I, outputs[j]);
        want[j] = falloc((int64_t)S * outputs[j]);
        owned[j] = falloc((int64_t)S * outputs[j] + 2);
        owned[j][0] = -177.f; owned[j][(size_t)S * outputs[j] + 1] = 211.f;
        p[j] = (Q36Projection){owned[j] + 1, &w[j], outputs[j], j == 1 ? 17 : 0};
    }
    if (format && count > 1) {
        QW *v = &w[1];
        if (format == 1) {
            float *raw = falloc((int64_t)I * v->O);
            for (size_t i = 0; i < (size_t)I * v->O; i++) raw[i] = sample_value(i);
            v->w = raw;
            free(v->q); free(v->sc); v->q = NULL; v->sc = NULL;
        } else if (format == 2) {
            v->h = malloc((size_t)I * v->O * sizeof(uint16_t)); assert(v->h);
            for (size_t i = 0; i < (size_t)I * v->O; i++) v->h[i] = (uint16_t)(0x3800 + (i * 17) % 0x700);
        } else {
            assert(I % 64 == 0);
            float *raw = falloc((int64_t)I * v->O);
            for (size_t i = 0; i < (size_t)I * v->O; i++) raw[i] = sample_value(i);
            v->q4 = malloc((size_t)I * v->O / 2); v->sg = falloc((int64_t)v->O * (I / 64));
            assert(v->q4); pack_int4_g64_planar(raw, v->q4, v->sg, v->O, I); free(raw);
        }
    }
    setenv("QWEN_PROJECTION_GROUP", "0", 1);
    counters_reset();
    for (int j = 0; j < count; j++) matmul_d(want[j], x, &w[j], S, I, outputs[j]);
    uint64_t original_quantizations = g_qwen_activation_quant_calls;
    /* The exact opt-out follows original per-matrix dispatch, too. */
    q36_projection_group(p, count, x, S, I);
    for (int j = 0; j < count; j++) assert(!memcmp(want[j], p[j].y, (size_t)S * outputs[j] * sizeof(float)));
    assert(g_qwen_projection_group_dispatches == 0);
    unsetenv("QWEN_PROJECTION_GROUP");
    g_qwen_activation_fail_alloc = heap_fail;
    g_qwen_activation_limit = cap_fail ? 16 : STRATA_ACTIVATION_MAX_BYTES;
    counters_reset(); q36_projection_group(p, count, x, S, I);
    int grouped = group_mode && !format && count >= 2 && !heap_fail && !cap_fail;
    assert(g_qwen_projection_group_dispatches == (uint64_t)grouped);
    assert(g_qwen_activation_quant_calls == (grouped ? (uint64_t)S : original_quantizations));
    size_t bytes = (size_t)S * (I + sizeof(float));
    assert(g_qwen_activation_heap_allocs == (uint64_t)(group_mode && !format && count >= 2 &&
            !cap_fail && bytes > STRATA_ACTIVATION_LOCAL_BYTES));
    for (int j = 0; j < count; j++) {
        assert(!memcmp(want[j], p[j].y, (size_t)S * outputs[j] * sizeof(float)));
        assert(owned[j][0] == -177.f && owned[j][(size_t)S * outputs[j] + 1] == 211.f);
        weight_free(&w[j]); free(want[j]); free(owned[j]);
    }
    free(x); g_qwen_activation_fail_alloc = 0; g_qwen_activation_limit = STRATA_ACTIVATION_MAX_BYTES;
}

static void core_group_gates(void) {
    enum {S = 5, I = 65, O = 17};
    int8_t x[S * I], q[O * I]; float sx[S], sc[O], y[S * O], wanted[S * O];
    for (int i = 0; i < S * I; i++) x[i] = (int8_t)((int)(next_random() % 255) - 127);
    for (int i = 0; i < O * I; i++) q[i] = (int8_t)((int)(next_random() % 256) - 128);
    for (int s = 0; s < S; s++) sx[s] = (float)(s + 1) / 91.f;
    for (int o = 0; o < O; o++) sc[o] = (float)(o + 1) / 73.f;
    IdotProjection p = {y, q, sc, O};
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        int32_t sum = 0; for (int i = 0; i < I; i++) sum += (int32_t)q[o * I + i] * x[s * I + i];
        wanted[s * O + o] = (float)sum * sc[o] * sx[s];
    }
    assert(matmul_q_idot_group(&p, 1, x, sx, S, I)); assert(!memcmp(y, wanted, sizeof y));
    float before[S * O]; memcpy(before, y, sizeof y);
    assert(!matmul_q_idot_group(NULL, 1, x, sx, S, I));
    assert(!matmul_q_idot_group(&p, 0, x, sx, S, I));
    assert(!matmul_q_idot_group(&p, 17, x, sx, S, I));
    assert(!matmul_q_idot_group(&p, 1, x, sx, 0, I));
    assert(!matmul_q_idot_group(&p, 1, x, sx, S, INT_MAX));
    assert(!matmul_q_idot_group(&p, 1, NULL, sx, S, I));
    IdotProjection alias = {(float *)q, q, sc, O};
    assert(!matmul_q_idot_group(&alias, 1, x, sx, S, I));
    IdotProjection duplicate[] = {p, p};
    assert(!matmul_q_idot_group(duplicate, 2, x, sx, S, I));
    IdotProjection bad = p; bad.O = -1;
    assert(!matmul_q_idot_group(&bad, 1, x, sx, S, I));
    assert(!memcmp(y, before, sizeof y));
    /* 16 descriptors, distinct output slices, still one safe dispatch. */
    float ys[16 * S * O]; IdotProjection many[16];
    for (int j = 0; j < 16; j++) many[j] = (IdotProjection){ys + j * S * O, q, sc, O};
    assert(matmul_q_idot_group(many, 16, x, sx, S, I));
    for (int j = 0; j < 16; j++) assert(!memcmp(ys + j * S * O, wanted, sizeof wanted));
}

static void shared_mlp_callsite(void) {
    enum {D = 65, F = 33, S = 5};
    Model m; Layer layer; memset(&m, 0, sizeof m); memset(&layer, 0, sizeof layer);
    m.c.hidden = D; m.c.shared_inter = F;
    layer.sh_g = weights(D, F); layer.sh_u = weights(D, F); layer.sh_d = weights(F, D);
    float x[S * D], want[S * D], got[S * D], g[F], u[F], down[D];
    for (int i = 0; i < S * D; i++) x[i] = sample_value(i);
    memset(want, 0, sizeof want); memset(got, 0, sizeof got);
    setenv("QWEN_PROJECTION_GROUP", "0", 1);
    qwen_shared_experts_cpu(&m, &layer, x, S, want, g, u, down);
    unsetenv("QWEN_PROJECTION_GROUP"); counters_reset();
    qwen_shared_experts_cpu(&m, &layer, x, S, got, g, u, down);
    assert(!memcmp(want, got, sizeof want));
    assert(g_qwen_projection_group_dispatches == (uint64_t)group_mode);
    assert(g_qwen_activation_quant_calls == (uint64_t)(integer_mode ? (group_mode ? 2 * S : 3 * S) : 0));
    memset(want, 0, sizeof want); memset(got, 0, sizeof got);
    setenv("QWEN_PROJECTION_GROUP", "0", 1);
    qwen_shared_experts_cpu(&m, &layer, x, 1, want, g, u, down);
    unsetenv("QWEN_PROJECTION_GROUP"); counters_reset();
    qwen_shared_experts_cpu(&m, &layer, x, 1, got, g, u, down);
    assert(!memcmp(want, got, D * sizeof(float)));
    assert(g_qwen_projection_group_dispatches == (uint64_t)group_mode);
    assert(g_qwen_activation_quant_calls == (uint64_t)(integer_mode ? (group_mode ? 2 : 3) : 0));
    weight_free(&layer.sh_g); weight_free(&layer.sh_u); weight_free(&layer.sh_d);
}

typedef struct {const QW *w; const float *x; const float *want; int ok;} ThreadCase;
static void *concurrent_group(void *arg) {
    ThreadCase *c = arg; enum {S = 3, I = 129, O = 17};
    float out[2][S * O];
    Q36Projection p[] = {{out[0], c->w, O, 0}, {out[1], c->w, O, 0}};
    c->ok = 1;
    for (int k = 0; k < 5; k++) {
        q36_projection_group(p, 2, c->x, S, I);
        if (memcmp(out[0], c->want, sizeof out[0]) || memcmp(out[1], c->want, sizeof out[1])) c->ok = 0;
    }
    return NULL;
}
static void concurrent_callers(void) {
    enum {S = 3, I = 129, O = 17};
    QW w = weights(I, O); float x[2][S * I], want[2][S * O];
    ThreadCase cases[2]; pthread_t threads[2];
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < S * I; i++) x[j][i] = sample_value((size_t)i + j * 31);
        matmul_d(want[j], x[j], &w, S, I, O);
        cases[j] = (ThreadCase){&w, x[j], want[j], 0};
    }
    /* Test-only counters would be shared mutable state; numerical workspaces
     * are independent. Disable instrumentation while callers run in parallel. */
    g_qwen_activation_count_enabled = 0;
    for (int j = 0; j < 2; j++) assert(!pthread_create(&threads[j], NULL, concurrent_group, &cases[j]));
    for (int j = 0; j < 2; j++) { assert(!pthread_join(threads[j], NULL)); assert(cases[j].ok); }
    g_qwen_activation_count_enabled = 1; weight_free(&w);
}

int main(int argc, char **argv) {
    integer_mode = !(argc > 1 && argv[1][0] == '0');
    group_mode = integer_mode;
#if !defined(__AVX2__) || defined(__AVX512VNNI__)
    group_mode = 0;
#endif
    setenv("COLI_DENSE_IDOT", integer_mode ? "1" : "0", 1);
    setenv("COLI_DENSE_BITS", "8", 1);
    prepared_activation_gates();
    const int rows[] = {1, 2, 3, 5, 17};
    const int columns[] = {1, 7, 15, 31, 32, 63, 64, 65, 129, 1024};
    for (int mode = 0; mode < 2; mode++) {
        setenv("COLI_IDOT_ROWS", mode ? "1" : "0", 1);
        core_group_gates();
        for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++)
            for (size_t i = 0; i < sizeof columns / sizeof columns[0]; i++)
                for (int count = 1; count <= 3; count++) one_group(rows[r], columns[i], count, 0, 0, 0);
        for (int format = 1; format <= 3; format++) one_group(5, 128, 3, format, 0, 0);
        one_group(17, 4097, 3, 0, 0, 0); /* bounded larger-row heap */
        one_group(17, 4097, 3, 0, 1, 0); /* allocation refusal preserves fallback */
        one_group(5, 129, 3, 0, 0, 1);   /* explicit preparation capacity refusal */
        shared_mlp_callsite(); concurrent_callers();
    }
    puts("strata shared activation: exact groups, quantization reuse, bounds, fallback and reentrancy ok");
    return 0;
}

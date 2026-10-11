/* Synthetic F32 head-bank microbenchmark. Includes shape/finite/alias checks.
 * These weights are untrained. This measures no model accuracy, verification,
 * GPU transfer, or end-to-end token throughput. Speedup may be below one.
 * Build with/without -fopenmp; set OMP_NUM_THREADS for repeatable comparisons. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../strata_heads.h"

static double bench_now(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static int bench_run(const strata_heads_bank *bank, const float *hidden,
                     float *workspace, size_t workspace_count, float *logits,
                     size_t logit_count, int *tokens, int parallel,
                     unsigned long iterations, double *seconds, unsigned long *checksum) {
    unsigned long i;
    double start = bench_now();
    for (i = 0; i < iterations; ++i) {
        if (strata_heads_predict(bank, hidden, bank->hidden_dim, workspace, workspace_count,
                                 logits, logit_count, tokens, bank->head_count, parallel) != STRATA_HEADS_OK)
            return 1;
        *checksum += (unsigned long)tokens[i % bank->head_count];
    }
    *seconds = bench_now() - start;
    return 0;
}

int main(int argc, char **argv) {
    enum { HEADS = 8, HIDDEN = 256, RANK = 32, VOCAB = 2048 };
    strata_heads_head heads[HEADS];
    strata_heads_bank bank = {HIDDEN, VOCAB, HEADS, heads};
    float *down = NULL, *up = NULL, *hidden = NULL, *workspace = NULL, *logits = NULL, *reference = NULL;
    size_t down_count = (size_t)HEADS * RANK * HIDDEN, up_count = (size_t)HEADS * VOCAB * RANK;
    size_t workspace_count = HEADS * RANK, logit_count = HEADS * VOCAB, i;
    int tokens[HEADS], reference_tokens[HEADS], exit_status = 1;
    unsigned long iterations = 100, checksum = 0;
    double serial_seconds, parallel_seconds, ignored;
    if (argc > 2) { fprintf(stderr, "usage: %s [iterations:1..100000]\n", argv[0]); return 1; }
    if (argc == 2) {
        char *end;
        errno = 0;
        iterations = strtoul(argv[1], &end, 10);
        if (errno || !argv[1][0] || *end || iterations < 1 || iterations > 100000) {
            fputs("iterations must be in 1..100000\n", stderr); return 1;
        }
    }
    down = malloc(down_count * sizeof(float));
    up = malloc(up_count * sizeof(float));
    hidden = malloc(HIDDEN * sizeof(float));
    workspace = malloc(workspace_count * sizeof(float));
    logits = malloc(logit_count * sizeof(float));
    reference = malloc(logit_count * sizeof(float));
    if (!down || !up || !hidden || !workspace || !logits || !reference) {
        fputs("benchmark allocation failed\n", stderr); goto cleanup;
    }
    for (i = 0; i < down_count; ++i) down[i] = (float)((int)(i % 31) - 15) * 0.002f;
    for (i = 0; i < up_count; ++i) up[i] = (float)((int)(i % 43) - 21) * 0.003f;
    for (i = 0; i < HIDDEN; ++i) hidden[i] = (float)((int)(i % 13) - 6) * 0.1f;
    for (i = 0; i < HEADS; ++i) {
        heads[i].future_offset = i + 2;
        heads[i].rank = RANK;
        heads[i].down = down + i * RANK * HIDDEN;
        heads[i].down_count = RANK * HIDDEN;
        heads[i].up = up + i * VOCAB * RANK;
        heads[i].up_count = VOCAB * RANK;
        heads[i].bias = NULL;
        heads[i].bias_count = 0;
    }
    if (strata_heads_predict(&bank, hidden, HIDDEN, workspace, workspace_count,
                             reference, logit_count, reference_tokens, HEADS, 0) != STRATA_HEADS_OK ||
        strata_heads_predict(&bank, hidden, HIDDEN, workspace, workspace_count,
                             logits, logit_count, tokens, HEADS, 1) != STRATA_HEADS_OK ||
        memcmp(reference, logits, logit_count * sizeof(float)) ||
        memcmp(reference_tokens, tokens, sizeof(tokens))) {
        fputs("serial/parallel correctness comparison failed\n", stderr); goto cleanup;
    }
    if (bench_run(&bank, hidden, workspace, workspace_count, logits, logit_count, tokens, 0, 3, &ignored, &checksum) ||
        bench_run(&bank, hidden, workspace, workspace_count, logits, logit_count, tokens, 1, 3, &ignored, &checksum) ||
        bench_run(&bank, hidden, workspace, workspace_count, logits, logit_count, tokens, 0, iterations, &serial_seconds, &checksum) ||
        bench_run(&bank, hidden, workspace, workspace_count, logits, logit_count, tokens, 1, iterations, &parallel_seconds, &checksum)) {
        fputs("benchmark prediction failed\n", stderr); goto cleanup;
    }
    printf("Synthetic untrained head bank: heads=%d hidden=%d rank=%d vocab=%d calls=%lu\n",
           HEADS, HIDDEN, RANK, VOCAB, iterations);
#ifdef _OPENMP
    printf("OpenMP max threads=%d; elapsed wall time; full validation included\n", omp_get_max_threads());
#else
    puts("Scalar fallback; process CPU time; full validation included");
#endif
    printf("serial: %.3f ms/call; parallel: %.3f ms/call; checksum=%lu\n",
           serial_seconds * 1000.0 / iterations, parallel_seconds * 1000.0 / iterations, checksum);
    puts("Microbenchmark only: no trained-head accuracy or model token-throughput result.");
    exit_status = 0;
cleanup:
    free(reference); free(logits); free(workspace); free(hidden); free(up); free(down);
    return exit_status;
}

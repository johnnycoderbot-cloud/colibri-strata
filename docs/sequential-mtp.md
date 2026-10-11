# Pretrained sequential MTP

The qwen36 CPU adapter now uses the checkpoint's actual trained MTP cell.
Drafts can run sequentially: each later guess uses the preceding MTP normalized
hidden output and token. Colibri verifies the proposed window with its full
target model and retains only the matching prefix.

The complete fork also measured **15.1% faster paired decoding than unchanged
Colibri**, winning all 15 pairs. Median arm rates were 40.827 and 45.997 tokens/s;
paired workload medians were +11.7%, +20.4% and +15.1%. This separate comparison
includes both the earlier CPU ports and pretrained MTP.
[Unchanged-stock comparison](../experimental/strata/results/pretrained-mtp-stock-comparison.json)

The one-draft paired trial measured **13.1% faster decoding**: median arm rates
41.241 → 46.657 tokens/s, with all 15 candidate pairs faster. The same optimized
binary ran both arms, with existing prompt lookup enabled. Three local test
prompts, five alternating pairs each, 64-token output cap, 8 OpenMP threads,
native Qwen3.5-0.8B INT8 weights/integer-dot activations, warm OS page cache.
All three separate greedy token/logprob/output comparisons matched exactly.
The paired full-request wall-speed median was 1.1221. This is a measured CPU
result for this checkpoint; it does not establish other hardware/model gains.
[Full paired measurements](../experimental/strata/results/pretrained-mtp-depth1-comparison.json)

## What changed

- The opt-in converter preserves all 15 pretrained MTP tensors in a separate
  directory. Existing target assets and their previous fingerprints stay intact.
- The native cell implements embedding-first projection, raw-weight-plus-one
  RMS norms, gated grouped-query attention, partial rotary positions, SwiGLU,
  residuals, final norm and the target's shared output projection.
- Strata-style KV-only prefill/catch-up prepares known MTP pairs without running
  their complete attention/MLP/output-head pass.
- Predicted hidden/KV rows stay provisional even when their guessed tokens match.
  The verifier's actual normalized target rows replace them through teacher-forced
  catch-up. Prefix reuse, snapshots and cancellation preserve that bookkeeping.
- The gate charges the first draft projection as well as later ones and measures
  a real plain-step time before judging whether drafting pays.

These follow the sequential-chain and KV-only scheduling ideas in
[pinned UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata/blob/d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf/src/core/mtp.cpp).
The checkpoint-specific layer math follows the
[official Qwen3.5 implementation in vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py).
Colibri's existing qwen38 MTP supplied the accepted-state/pending-row design;
the HC4 Flash layer geometry is not reused for this single-stream dense model.

## Existing checkpoint setup

The first qualified mode is dense qwen36 CPU INT8/IDOT, one KV slot, shared
embedding/readout and one pretrained MTP layer. The C numerical tests also have
an F32 cell reference. Packed INT4/INT16 targets, active device execution, MoE
sidecars, sampled learned proposals and other family adapters need separate
qualification. The existing independent heads remain an opt-in research path.

```sh
python c/tools/convert_qwen36.py --mtp-only --model SOURCE_CHECKPOINT \
  --native-target NATIVE_MODEL --out NEW_MTP_DIR \
  --native-precision 'qwen36:DBITS=8;IDOT=1;kernel=cpu;tap=post-final-norm-mtp/v1'
make -C c qwen36 strata-native-check
c/coli run --model NATIVE_MODEL --gpu none --mtp-dir NEW_MTP_DIR \
  --mtp-drafts 1 --temp 0 'Explain CPU caches.'
```

Sidecar conversion is explicit, preserves source/native assets, refuses existing
output directories, and binds exact target bytes/precision/tap to the sidecar's
payload digest. Default target conversion still skips MTP. Depths 2/3 are
implemented and passed actual model-output/hidden-state correctness checks;
the current performance claim is for depth 1 only.

```sh
python c/tools/native_optimization_bench.py --baseline BASELINE_ENGINE \
  --optimized c/qwen36 --model NATIVE_MODEL --prompts three-test-prompts.json \
  --output NEW_RESULT.json --threads 8 --tokens 64 --repeats 5 \
  --studies mtp --mtp-dir NEW_MTP_DIR --mtp-drafts 1
```

The `mtp` study uses the optimized executable in both arms; the baseline binary
argument is retained by the common collector but is not that study's control.
Select `--studies stock-mtp` for the complete fork against that unchanged
baseline executable.
Force/oracle/trace settings are cleared. The report requires evidence of actual
MTP calls, rejects cell failures, and verifies all asset/binary hashes afterward.
MTP call time is source-only; the shared verifier counters also include lookup.

The pretrained-cell segment test scored 194/279 correct first guesses (69.5%),
using captured generation rows without their earlier prompt MTP cache. It is
separate from full-prefix native correctness and performance. Real-weight kernel
oracle error stayed below 1.55e-5 for F32 and 4.77e-6 for native INT8.
[Segment reference](../experimental/strata/results/pretrained-mtp-segment-quality.json),
[native kernel oracle](../experimental/strata/results/pretrained-mtp-kernel-oracle.json),
[actual depth1/2/3 model parity](../experimental/strata/results/pretrained-mtp-actual-frontier.json),
[real-model rollback and refusal cases](../experimental/strata/results/pretrained-mtp-rollback-frontier.json).
The latter uses explicit forced proposal IDs for acceptance/rejection control;
it proves correctness while executing the authentic cell, not predictor quality
or speed. Its active cases include first/middle/cyclic rejection, full acceptance,
budget clipping, pin restoration, sampling fallback, EOS and cancellation with
subsequent continuation.

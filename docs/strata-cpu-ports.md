# Strata CPU optimizations inside Colibri

Colibri stays the execution engine. The requested reference is
[UnbiasedStrata at d0a44ea](https://github.com/robertkarl/UnbiasedStrata/tree/d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf),
whose engine files retain Strata 0.1.39. The MIT-licensed code informed these
equivalent Colibri implementations; the reference engine's sequential MTP chain
is excluded from the independently drafted future-token heads.

| Reference optimization | Colibri implementation | Current qualification |
|---|---|---|
| Prepared activations shared across projections (`ActQ` and `ThreadPool` callers) | Caller-owned bounded INT8 activation scratch shared by QKV, MLP gate/up and DeltaNet qkv/z; one output-row OpenMP dispatch | qwen36 INT8 CPU projection groups on qualified AVX2/VNNI; other paths retain their existing drivers |
| Weight-block and sign preparation shared across token rows (`row_dot_multi_z`) | `dot_i8i8_rows` reuses those operations for 2–4 independent verification rows | Exact scaled-float parity against previous rowwise INT8 CPU dots; S=1 retains old default path |
| Wider/unrolled integer accumulation | Optional `dot_i8i8_wide` with four accumulators | `COLI_IDOT_WIDE=1` opt-in, default off; microbenchmark gains vary by matrix and thread count |

Reference locations:
[activation/pool](https://github.com/robertkarl/UnbiasedStrata/blob/d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf/include/strata/kernels/cpu/pool.hpp),
[expert multirow dot](https://github.com/robertkarl/UnbiasedStrata/blob/d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf/src/kernels/cpu/expert.cpp),
[AVX2 preparation](https://github.com/robertkarl/UnbiasedStrata/blob/d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf/src/kernels/cpu/kq_avx2.cpp).

Activation preparation preserves Colibri's quantizer, per-row scale and reduction
order. Scratch is explicit and belongs to the caller; no address-based reuse can
silently borrow another call's activations. The small path uses local scratch;
larger groups use a bounded allocation and fall back to established projections
on allocation failure or unsupported layouts. Source-only signed INT4/INT16
head-bank codecs do not qualify native INT4/INT16 target arithmetic here.

Use `QWEN_PROJECTION_GROUP=0` and `COLI_IDOT_ROWS=0` for independent ablations.
ARM/i8mm, AVX512VNNI and active GPU routes retain their established kernels. New
accelerator execution and other model families require their own real tests.

On the tested Qwen3.5-0.8B CPU model, resident dense weights mean warm decoding
cannot benefit from SSD expert prefetch. Colibri already has model-specific
streaming/cache paths. These two ports do not add a new universal SSD-to-device
pipeline, hot-expert residency policy or compressed KV implementation.

The actual all-port trial records a 1.0129 median paired decode ratio; isolated
row/group effects are near noise. The existing parallel heads made that engine
slower. [Full measurements](../experimental/strata/results/unbiased-cpu-optimizations.json)
The subsequent independent prompt set measures 1.0331 for the CPU ports and
0.9706 when adding the new residual heads. The gain remains small and varies
by workload; the parallel-head bank still costs time overall.
[Fresh measurements](../experimental/strata/results/residual-fresh-cpu-comparison.json)
The wide kernel stays off by default: the eight-thread vocabulary projection
microbenchmark was approximately unchanged, despite gains on a small cache-hot
FFN matrix. Kernel timings are not model tokens per second.

Reproduce against a caller-preserved unchanged executable:

```sh
make -C c qwen36 strata-native-check
python c/tools/native_optimization_bench.py --baseline BASELINE_ENGINE \
  --optimized c/qwen36 --model MODEL_DIR --prompts three-prompts.json \
  --output new-result.json --threads 8 --tokens 64 --repeats 5 \
  --rows 1 --projection-group 1 --wide 0 --ablate-knobs
```

Add `--heads BANK --head-input next-embedding` to measure the bank separately.
The `--studies` selector can isolate stock, rows, group, wide and heads. Every
study checks greedy numeric DATA/token parity separately from timing, records
asset/binary hashes, charges the entire bank once and retains failures. Asset
hashing warms OS page cache; fresh process startup is a separate observation.

# Benchmark protocol

The draft-depth trials retain maximum one: maximum two measured +1.3% overall
against it, maximum three +0.1%, with workload regressions in both studies.
[Depth evidence and choice](sequential-mtp.md#measured-draft-depth-choice).
Existing native OLMoE CPU prefetch measured -1.4% with eight cached experts per
layer and +0.6% with sixteen; these warm-filesystem whole-request results keep
prefetch opt-in. [MoE trial, exact parity and timing scope](moe-prefetch-benchmark.md).

The complete CPU fork versus unchanged Colibri measures **1.1508 median paired
decode ratio (+15.1%)**, with all 15 candidate pairs faster. Per-prompt paired
medians are 1.1168/1.2044/1.1508; median arm rates 40.827 → 45.997 tokens/s.
This comparison includes CPU ports plus pretrained sequential MTP. Same model,
hardware, precision,64-token cap,lookup and alternating protocol as below.
[Unchanged-stock comparison](../experimental/strata/results/pretrained-mtp-stock-comparison.json)

The new pretrained sequential MTP trial achieves **1.1313 median paired decode
ratio (+13.1%)** against the same optimized binary with MTP disabled. Median
arm rates 41.241 → 46.657 tokens/s, all 15 candidate pairs faster, per-prompt
paired medians 1.0735/1.1367/1.1373. Same native Qwen3.5-0.8B INT8/IDOT target,
8 threads, lookup enabled, three local test requests, 64-token cap and five
alternating pairs. Three separate numeric comparisons match exactly; measured
draft/verification costs are included and no forced guesses run during timing.
[Measurements](../experimental/strata/results/pretrained-mtp-depth1-comparison.json)
[Implementation and reproduction](sequential-mtp.md)

The subsequent fresh-prompt trial confirms a small CPU-port gain: **1.0331
median paired decode ratio**, with heads off. Per-prompt medians are 1.0331,
1.0720 and 1.0044. The same binary with the new rank24 residual bank enabled
measures **0.9706 (2.9% slower)**. Six separate greedy numeric correctness
requests match. The three new requests differ from all training/development
prompts; the other settings remain 8 threads, INT8/integer-dot, prompt lookup,
64-token cap and five alternating pairs per prompt. Model/binary/bank hashes
were stable. [Fresh measurements](../experimental/strata/results/residual-fresh-cpu-comparison.json)

The earlier isolated CPU-port trial measures a **1.0129 median paired decode
ratio** against unchanged Colibri with heads disabled. The three prompt-level
median ratios are 1.0357, 1.0029 and 1.0019: a small workload-dependent gain,
not a major or universal speedup. Shared activation/projection grouping and INT8
verification-row weight reuse were tested separately; their paired medians are
1.0028 and 1.0047. Turning on the existing rank16 parallel bank against the same
optimized binary gives **0.9779**, slower. All twelve separate numeric correctness
requests match. Same Qwen3.5-0.8B INT8/integer-dot target, 8 threads, three prompts,
five alternating pairs each,64-token output cap, existing prompt lookup in every
arm. Warm page cache; no isolated prefill duration is available.
[Measurements and hashes](../experimental/strata/results/unbiased-cpu-optimizations.json)
[Port scope and source](strata-cpu-ports.md)

A real native comparison now exists, with **no clear repeatable speedup**:
unchanged Colibri 39.829 vs the two-head CPU fork 39.471 median tokens/s across
15 interleaved pairs, identical outputs. Both use Qwen3.5-0.8B, INT8/integer-dot,
8 CPU threads and existing prompt lookup. The candidate won eight pairs and lost
seven. [Raw measurements and identity](../experimental/strata/results/native-cpu-comparison.json)

Strata was not run locally: this Intel UHD machine has no supported Strata
setup/backend, and Strata does not accept the small dense checkpoint. Its
published Qwen3.8-Flash-Next Q2_0 GPU references are 93.0 tokens/s at 4K and 73.7
at 128K on an RTX5070. These are a separate model/hardware reference, not fair
ratios against our CPU run. [Pinned Strata measurements](https://github.com/Niko1221/Strata/blob/61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406/docs/DETAILS.md)

The head microbenchmark remains a synthetic F32 head-bank measurement rather
than model tokens/s. Controlled oracle-biased native banks prove rollback only.

Use the existing `c/tools/benchmark_http_serving.py` / `benchmark_baseline.py`
collectors where suitable. For each ablation save raw logs plus a manifest with:

| Required identity | Required measurement |
|---|---|
| Engine commit; model revision and artifact hashes; tokenizer/template digest | TTFT; prefill tokens/s; steady decode tokens/s; full request time |
| Head pack hash, hidden tap, offsets, rank, training corpus/revision | Acceptance conditional on earlier heads, proposals/accepted by offset, target forwards |
| CPU/GPU, RAM/VRAM, OS/driver, PCIe topology and NVMe | Peak allocated/resident RAM/VRAM; CPU/GPU work and transfer stall/overlap |
| Prompt/output token hashes, seed, temperature, penalties, reasoning policy | Greedy token/session parity where asserted; held-out quality for lossy KV |
| Prompt/context length, chunk size, output cap, cold/warm cache policy | Expert hit/miss/promotion, demanded/prefetched/unused bytes, p50/p95 latency |

Interleave baseline and feature runs, with at least five measured paired repeats
after explicit warmups. Use the same checkpoint/precision/backend and test one
change at a time: heads, prefetch, popularity, chunked prefill, KV; then combined.
Count every drafting, verification, replay, transfer and synchronization cost.
Separate baseline target forwards from the extra reference replay forwards.

Sweep context lengths 4K/32K/64K/128K only where the model supports them, and short
versus long output. Greedy parity and sampled distribution/seed reproducibility are
different tests. KV quantization needs a quality report in addition to throughput.

Performance claims must include hardware, quantization, context, workload and
acceptance. A microbenchmark, synthetic perfect guesses or a successful compile
cannot justify a 53–200 tokens/s claim or universal zero context slowdown.

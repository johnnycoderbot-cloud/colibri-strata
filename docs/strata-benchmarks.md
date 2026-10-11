# Benchmark protocol

No real-model speedup has been measured for the new fork components. The head
microbenchmark measures a synthetic F32 head bank only; it is not model tokens/s.

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

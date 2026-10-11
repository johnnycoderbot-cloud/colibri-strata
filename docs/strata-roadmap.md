# Integration roadmap

Portable-core tests are the starting gate. Each task below remains pending until
its stated evidence exists; model-specific trained heads are separate artifacts.

## 1. Real head adapters across the text engines

Start with the existing verifiers in GLM, qwen36/qwen38 and DeepSeek; next add
OLMoE's all-row readout, then hybrid and sliding-window families. Every adapter:

- Identifies the exact checkpoint revision, tokenizer digest, hidden tap and
  quantization; refuses mismatched packs even when dimensions coincide.
- Captures the immutable hidden vector from the same target pass used for offset1,
  predicts contiguous future offsets independently, and passes them to the target.
- Snapshots KV, position metadata, rings, recurrence, convolution and device mirrors.
  Rejected rows must leave no persistent mutations. Tests force rejection at every
  depth, EOS, stop, token budgets, prefix reuse and allocation/backend failures.
- Uses identical numerical kernels for greedy plain/verify rows where exact parity
  is promised. Sampled requests stay on the existing path until an exact sampling
  algorithm and RNG contract are tested.
- Measures **total bank cost once per prediction**. Parallel heads must not be
  charged as sequential `(depth-1) * draft_cost` steps by the adaptive gate.

Exit evidence: per-family tiny oracle, exact greedy continuation/session state,
real checkpoint held-out acceptance by offset, trained-pack identity and benchmark
against that same engine with heads off. Batched prefill alone does not pass.

## 2. DMA double buffering and chunked prefill

Bind the two-slot lifecycle to existing expert stores and CUDA/HIP transfer streams.
Lookahead only suggests experts; actual routing determines the demand set. Retain
a synchronous fallback for missing predictions, allocation failure and device loss.
Do not reuse or evict a buffer before both copy and consumer events finish.

Use per-device workspace budgeting to sweep chunk sizes 512/1024/2048/4096/6144/8192
where they fit. Transfer telemetry must distinguish predicted/used/wasted bytes,
stall time, copy overlap and peak VRAM. Do not force Strata's Qwen-specific chunk
size or memory assumptions on every family.

Exit evidence: fake event lifecycle tests plus real GPU event timeline, routed expert
equivalence, fallback parity, TTFT and prefill tokens/s across prompt/context lengths.

## 3. Adaptive hot-expert residency

Merge decaying popularity into the existing placement/cache policy. Reserve dense
weights, KV, workspace and staging buffers before selecting experts. GPU promotion
and eviction require fences; CPU fallback handles cold experts. Include per-layer
fairness and avoid thrashing/one token's demand evicting active consumers.

Exit evidence: trace-driven cache hit/miss/bytes and bounded peak VRAM; fixed router
and precision; cold/warm benchmarks and promotion overhead; no fake OS memory lock.

## 4. Model-native INT8 KV kernels

Map each engine's attention layout to block quantization and dequantization kernels.
Attention-only, MLA, sliding-ring and sparse attention layouts need distinct adapters.
Recurrent DeltaNet/KDA state is not an ordinary attention KV cache. Preserve RoPE,
positions, window eviction, snapshots and rejected draft rows.

Exit evidence: memory accounting at 4K/32K/64K/128K (only within model limits),
teacher-forced log-probability deltas, long-context retrieval, perplexity/task quality
and decode latency versus uncompressed state. Quantized KV can change greedy tokens;
do not call it lossless or enable by default before quality evaluation.

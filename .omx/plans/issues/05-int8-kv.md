The INT8 block codec is a tested lossy vector reference, not an integrated model
KV cache or attention kernel. Add model-native layouts/kernels and quality gates.

Acceptance:

- Distinct adapters for dense attention, MLA, sliding rings and sparse attention;
  do not quantize DeltaNet/KDA recurrence as if it were attention KV.
- Preserve RoPE, positions, window eviction, prefix persistence and speculative rollback.
- Account for bytes and overhead at supported 4K/32K/64K/128K contexts.
- Compare teacher-forced logprob/perplexity deltas, retrieval/task quality and latency
  against uncompressed state. INT8 can change greedy outputs; it is not lossless.
- Test allocation/backend failures and retain full-precision fallback.
- Enable by default only after held-out quality and end-to-end performance evidence.

See c/strata_memory.h, docs/strata-roadmap.md and docs/strata-benchmarks.md.

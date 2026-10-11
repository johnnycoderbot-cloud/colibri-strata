# Unbiased Strata optimizations inside Colibri

User clarified on 2026-10-11: the original objective is to add Strata advantages
to Colibri. Colibri remains the runtime; no separate Strata CPU engine is being
built. Future-token heads must draft independently in parallel. "Paired runs"
in the user's clarification meant parallel future-token drafting, not benchmark
methodology. Performance verification still requires actual paired measurements.

Requested source: robertkarl/UnbiasedStrata,
commit d0a44ea7e757c59fa35a78b4b98d0c4bb44ce2cf. Its engine source is unchanged
Strata 0.1.39 at 6f32ec070f23ced9f50e704d854d775da52591ab. Its MTP loop is
sequential too and is excluded from the new independent-head implementation.

Unchanged Colibri profile on the real Qwen3.5-0.8B native INT8/integer-dot model,
8 CPU threads, lookup off, 127 fed decode tokens: DeltaNet 7.71 ms/token,
attention 1.60, dense MLP 8.61, LM head 7.30, phase total 25.23. This diagnostic
is not a speedup comparison. Dense resident weights mean SSD expert prefetch
cannot improve this small model's warm decode.

Slices and real ownership:

- /root/memory_primitives: share one prepared activation across adjacent CPU
  QKV, gate/up and DeltaNet qkv/z projections. Bounded explicit scratch;
  one OpenMP dispatch where eligible; preserve rounding, scales and device
  fallback. Own scoped qwen36 changes, new activation helper/tests and only
  the new projection-group append in idot.h.
- /root/parallel_heads: shared INT8 weight-block/sign preparation across 2/4
  verification rows. Own dot_i8i8_rows and existing matmul_q_idot x86 dispatch,
  plus dedicated exactness/microbenchmark tests. Keep S=1 unchanged unless
  independently qualified. Coordinate helper ABI with the projection group.
- /root/architecture: benchmark tool with independent ablation flags, exact
  numeric/output checks and honest failure artifacts; subsequent review.
- /root: build wiring, source attribution, native rebuild, isolated real-model
  benchmarks, publication and durable status.

Acceptance: strict scalar/AVX2/VNNI and sanitizer exactness, tails/canaries,
allocation/fallback and concurrent-workspace isolation. Then unchanged Colibri
versus optimized Colibri without new heads (stock lookup enabled), >=5 paired
repeats on each of three held-out prompts, fixed64-token cap. Benchmark heads
separately against the same optimized engine to distinguish kernel and learned
head effects. Never describe a microbenchmark as model tokens/s or claim the
weak 12.9%/8.0% heads already accelerate decoding. Actual other-family/device
qualification and native MoE SSD/RAM/VRAM scheduling remain in scope later.

Completed real measurement: all-port paired median1.0129; row1.0047;
group1.0028; existing rank16 bank isolated0.9779 (slower). All12 numeric
correctness requests matched. New wideS1kernel remains optional/defaultoff:
8-thread vocabulary projection microbench ratio1.0009, no practical gain.

Next slice: explicit CSNHv2 independent residual heads. Training-only factors
reuse immutable raw native LM logits plus horizon-specific low-rank deltas;
v1 APIs refuse residual banks. PrototypeheldoutINT8top1 .2473/.07246,
bothprefix .03584, meanacceptedextra .28315. Runtime frontier caches must
match hidden/session/position before any sampler mutation; accepted verification
rows update both hidden and raw logits. Loader/exporter ownerparallel_heads,
runtime/cache ownerroot, realfrontierharness/review ownerarchitecture.
Prototype quality is not model acceleration; runisolatedsamebinaryheadsoff/on.

Final current slice: strict/scalar/OpenMP/sanitizers and13interoptests passed;
all live residual frontier gates and six native residual scorer replay states
passed. Combined native gate8C+62Python passed; dependencies8 andVKlink2 passed.
Fresh3prompt*5pairs: CPUports pairedmedian1.0331; new residualheads samebinary
0.9706 (slower). Actual gate-selected first-extra guesses3/31 in one repeat
perprompt, zero offset3 judged. This is not fulltraceaccuracy; no headspeedup.
Overall all-family/device throughput objective stays incomplete.

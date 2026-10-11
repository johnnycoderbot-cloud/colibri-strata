# Native throughput integration

User reaffirmed primary objective on 2026-10-11: actual measured faster native
Colibri inference across all families, not component/API/device-only milestones.
Prior bootstrap-only non-goals do not restrict this newly authorized integration.
Local model downloads and local head training needed for verification are in scope;
no paid/cloud training services are authorized or needed.

Prepared OLMoE, a native-supported real MoE that fits local RAM. Pin official
OLMoE-1B-7B-0125-Instruct GGUF revision2ac1d27317927518c9ef7bd99f91f3f1e4ee288d,
Q4_K_M source SHA2564ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f.
Convert with the existing native merged-I8 converter, preserve the exact container
and unchanged baseline binary; before/after use identical target tensors/precision.

Audit selected native qwen36 with real Qwen3.5-0.8B for the first live verifier
integration; it already has exact all-row verification and recurrent rollback.
Pinned source2fc06364715b967f1860aea9cf38778875588b17, native tied output head
materialized only under explicit source tying. Both real checkpoints are prepared.
The source hashes and baseline binaries are retained in ignored local directories.

Current real evidence: baseline coherent native answer ~30tokens/s,2232 target
hidden rows captured. First rank8 offset2 head trained400steps has low top1/acceptance;
three heldout prompts times5 paired runs retain IDs/logprobabilities but ratio0.975.
This is not a speedup. Two-head rank24 known-next-token training was unstable and
rejected; normalized rank16 training now gives heldout top1 12.9%/8.0%, both-prefix
3.23%, still substantial overfit. Exact native paired timing is the next gate;
pretrained output transfer is being prototyped separately. In both banks,
all heads read the same immutable concat(h_t,E[target-selected offset1]). The target
has already selected that token; no future target states or previous head predictions
are features. Distinct tap/precision fingerprint binds captures and trained exports.
Numeric modes outside default INT8+integer-dot CPU are refused until separately
fingerprinted/qualified. All-family goal remains incomplete. Active-bank real-target
rollback tests cover first/middle rejection, full acceptance/another proposal,
budget clipping, pin restore, sampled fallback, EOS and cancellation, with exact
token/logprob/hidden-frontier parity. Controlled oracle-biased banks are strictly
correctness evidence, never learned acceptance or throughput evidence.

Ownership / slices:

1. /root/architecture: read-only native hidden tap, all-row verify, KV rollback,
   numerical parity and protocol map. Native in-session role, no detached team.
2. /root/memory_primitives: independent read-only native cache/kernel opportunity
   and hardware/model feasibility; no overlapping state-verifier exploration.
3. /root: prepare real model, baseline binary, paired benchmark/provenance harness,
   choose concrete integration after those maps, maintain rollout and publication.
4. /root/parallel_heads: subsequent assigned pack/training slice after ABI is fixed.

Acceptance for first live adapter: learned independent future heads read the native
target hidden vector; exact checkpoint/tokenizer/quantization identity; causal
target verification; commit accepted KV without reference replay; cancellation,
EOS/output budgets/prefix rollback and allocation failure tests. Same numerical
rules as greedy baseline where exact equivalence is promised. Sampling retains
existing route until its distribution/RNG contract is proved.

Acceptance for speed: interleaved >=5 warmed paired runs, same real model/hardware,
same prompt/output caps; decode and prefill throughput/TTFT/full wall time, emitted
token hashes, actual proposals/acceptance and overhead, residency/transfer evidence.
Neither synthetic perfect guesses nor compile/GPU execution counts as acceleration.
An integration that loses speed is optimized or gated; document results honestly.

All families, actual GPU/NPU/TPU/AIPU paths and SSD/RAM/VRAM scheduling remain the
overall objective. First family success must not be recorded as universal completion.

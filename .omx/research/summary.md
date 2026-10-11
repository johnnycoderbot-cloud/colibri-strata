# Source findings

Pinned Colibri base: bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850.
Pinned Strata reference: 61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406.

The all-family scope has ten text engines, one image engine and two decision
engines. Image/decision token speculation is N/A. `docs/strata-fork.md` records
the complete matrix. Existing GLM/Qwen/DeepSeek verifiers are the first adapter
targets. OLMoE lacks all-row logits but has simpler attention state. Other families
need recurrent/convolution/sliding-window snapshots, not just a prefill batch.

Strata uses sequential native MTP drafting plus batched checking. Independent
parallel future heads require additional per-checkpoint trained weights. Current
Strata uses adaptive expert residency, CPU computation of uncached experts and
overlapped prefill transfers. Prefetch routing is predicted, not guaranteed.
INT8 KV is optional and lossy; context speed is not constant. Fixed Qwen layout,
prefill chunk ceiling and published hardware results are not portable promises.

Implementation is original Apache-2.0 code; no Strata source/weights copied.
Native architecture review found and repaired numeric hash identity admission and
JSON-versus-C Unicode escaping. Memory and head finite checks reject fast-math.
Verifier capacity admits eight learned offsets plus target offset1; engine adapters
must honor smaller actual batch limits. No production adapter is certified.

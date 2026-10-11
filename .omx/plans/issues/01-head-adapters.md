The portable `c/strata_heads.h` core currently runs independently of production
engines. Connect trained future-offset proposals to every text family while
preserving the target model's decisions and existing backend guards.

Coverage: glm, glm53, inkling, kimi, olmoe, qwen36, qwen38, deepseek_v4,
deepseek_v41, mimo; image and decision scoring are explicitly N/A.

Acceptance:

- Validate checkpoint/tokenizer/hidden-tap/quantization identity before attachment.
- Capture immutable target hidden state, retain target offset1, and predict
  contiguous offsets2+ independently with parallel head kernels.
- Reuse existing GLM/Qwen/DeepSeek batch verifiers; add all-row logits and safe
  snapshots for the remaining families, starting with attention-only OLMoE.
- Rejection at every depth, EOS, budgets, stops, prefix persistence and backend
  failures match plain greedy generation and committed KV/recurrent/device state.
- Preserve unsupported-backend fallbacks. Measure total head-bank cost once;
  do not apply the sequential per-depth MTP cost formula to parallel heads.
- Mark each family integrated only after a tiny oracle and held-out real-model
  acceptance/cost evidence. Sampled decoding remains on its original path.

See docs/strata-fork.md and docs/strata-roadmap.md. Current tests certify the
portable reference and synthetic target, not these engine adapters.

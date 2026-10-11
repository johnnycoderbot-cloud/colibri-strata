# Verification map

- C heads: finite/bounds refusal, independent future offsets, scalar/OpenMP parity,
  toy causal target parity for all/first/middle rejection, EOS, token budgets,
  snapshot/restore/commit failure behavior.
- C memory: INT8 finite roundtrip and error bound, zeros and capacity failures,
  budget/decay expert selection, fake async transfer and consumer event fences.
- Offline training: synthetic labeled hidden traces, strict checkpoint/tokenizer
  identity, future-offset labeling, export dimensions and compiled C predictions.
- Builds: GCC in existing kali-pc WSL; -Wall -Wextra; scalar and OpenMP; ASan+UBSan.
- Regression: existing standalone c/tests/test_spec_draft.c.
- Publication: inspect exact staged files, GitHub fork parent/HEAD, CI conclusion.
- Not certified: production adapters, real checkpoint acceptance, KV quality,
  CUDA/HIP DMA overlap, GPU hot-set promotion and end-to-end tokens/second.

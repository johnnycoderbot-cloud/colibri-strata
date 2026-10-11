# Execution ledger

- 2026-10-10: Read user OMX instructions and ultrawork/deep-interview/plan/team/
  executor skills. Started real OMX session. User resolved scope: all families.
- Cloned Colibri and Strata source at pinned revisions in requirements.
- Authenticated GitHub account: johnnycoderbot-cloud. No existing Colibri fork
  found in first 100 owned repositories. Available verification compiler: GCC
  13.3.0 in kali-pc WSL; Windows Python has torch/numpy/safetensors/transformers.
- Native architecture, parallel_heads, memory_primitives workers launched with
  disjoint ownership. OMX detached team runtime is not active.
- Discovery: both projects already have speculation. Strata's native MTP drafts
  autoregressively; independent multi-offset heads are additional model-specific
  training, not a copy of a universal existing Strata feature.
- Portable core complete: GCC 13.3, OpenMP and scalar tests; ASan+UBSan; existing
  test_spec_draft regression; 8 Windows Python tests including actual torch training.
  WSL does not have torch, so its Python training skip was NOT counted as a pass.
- Synthetic training/export compiled under GCC and both execution modes produced
  the trained expected tokens, including UTF-8/quoted metadata export after review.
- Native independent review verdict: approved for documented experimental scope.
  Corrected string-only hash admission, safe UTF-8 C escaping, finite-check guards,
  and verifier depth capacity. Production and hardware follow-ups remain pending.
- Created https://github.com/johnnycoderbot-cloud/colibri-strata; GitHub confirms
  `isFork=true`, parent JustVugg/colibri. Set local origin to fork, upstream to base.

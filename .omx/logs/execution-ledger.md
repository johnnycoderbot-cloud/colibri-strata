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
- Published code commit e910c19acb1f6be251e26bd9790b36858e78e4d4 on main.
- GitHub focused CI https://github.com/johnnycoderbot-cloud/colibri-strata/actions/runs/38101880229
  completed successfully: portable OpenMP/scalar/regression/sanitizers and actual
  CPU torch training plus compiled C export. Bootstrap verdict: approved.
- Created issues 1..5 for all-family adapters, real head capture/training, GPU
  prefetch/chunked prefill, hot residency and model-native INT8 KV quality.
  These production tasks remain pending; no new live inference speedup claimed.
- New request: INT4/INT8/INT16 and adversarial GPU/NPU/AIPU/NVMe testing; user
  identified Axelera and requested generic NPU/GPU/TPU interfaces.
- Added true signed CSQ1 codec, quantized independent heads, CPU/ONNX/JAX graph
  paths, safe bounded file tier and native installed Metis artifact check.
- 1.3M seeded ASan/UBSan cases passed. Cross-language INT4/8/16 bytes and tokens
  agree. Physical Intel UHD770 DML profile/parity and 64MiB NVMe hash reads passed.
- Found/fixed noncanonical Python codec acceptance, FIFO open hang, file completion
  admission race, UTF8 SDK output failure, SDK status/summary handling.
- Native Metis1.8.0 check passed1346/1346 vendor ResNet50 golden frames, without
  driver/firmware/reset/power changes. This does not certify Colibri head execution
  or native INT4/INT16 on Metis. No physical TPU available; refusal/CPU paths tested.
- Full upstream portable C suite running; independent final review and GitHub CI pending.
- Full portable C suite now passes; sole initial failure was the SSD vector test's
  checkout CRLF handling, corrected without weakening payload grammar validation.
- Independent final verdict approved. Fixed accelerator CLI pack admission and
  exclusive report/diagnostic creation; deterministic completion-race regression added.

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
  identified AIPU and requested generic NPU/GPU/TPU interfaces.
- Added true signed CSQ1 codec, quantized independent heads, CPU/ONNX/JAX graph
  paths, safe bounded file tier and native installed AIPU artifact check.
- 1.3M seeded ASan/UBSan cases passed. Cross-language INT4/8/16 bytes and tokens
  agree. Physical Intel UHD770 DML profile/parity and 64MiB NVMe hash reads passed.
- Found/fixed noncanonical Python codec acceptance, FIFO open hang, file completion
  admission race, UTF8 SDK output failure, SDK status/summary handling.
- Native AIPU1.8.0 check passed1346/1346 vendor ResNet50 golden frames, without
  driver/firmware/reset/power changes. This does not certify Colibri head execution
  or native INT4/INT16 on AIPU. No physical TPU available; refusal/CPU paths tested.
- Full upstream portable C suite running; independent final review and GitHub CI pending.
- Full portable C suite now passes; sole initial failure was the SSD vector test's
  checkout CRLF handling, corrected without weakening payload grammar validation.
- Independent final verdict approved. Fixed accelerator CLI pack admission and
  exclusive report/diagnostic creation; deterministic completion-race regression added.
- Broader inherited CI exposed my redundant handwritten header prerequisites in
  the new targets (test_makefile_deps); switched them to the existing automatic
  dependency mechanism and added that upstream gate to focused CI.
- All 8 generated-dependency checks passed after rebuilding the earlier manual
  benchmark through Make (creating its missing .d). Full C suite: 196 binaries.
- Focused CI run38103395741 passed including 270648 coverage-guided libFuzzer
  executions. OpenVINO native route subsequently added: actual CPU/GPU source
  precisions4/8/16 and execution placement passed; local NPU absent and refusal tested.
- User requested generic NPU/TPU/AIPU public names; renamed the AIPU adapter and
  made installed SDK entry/environment explicit. No vendor name in public labels.
- User raised speech/translation/camera expansion. Documented model-family and
  streaming-runtime design; those workloads are not falsely marked implemented.
- Final code31707104752f5c6961418f775eb366c2f7be0a9a verified by focused CI
  run38104321143, all jobs passed: generated-dependency regression, OpenVINO native
  CPU paths, training/export, precision tests, seeded sanitizer campaign and libFuzzer.
- Independent latest review approved. Local Windows42checks passed41 with POSIX
  FIFOinapplicable; actualLinuxFIFO/race6checks passed. NativeOpenVINOGPU+CPU
  allsourcebits4/8/16 physical placement/parity passed; no physicalNPU/TPU available.
- Precision/hardware hardening slice complete; speech/translation/camera runtime
  implementation remains tracked separately at GitHubissue6.

- Streaming expansion: user selected English, camera/image/video upscaling, text
  inference and actual accelerator execution. Three production graph families
  now use the existing registry, CLI and authenticated gateway with OpenVINO.
- Native in-session ownership: /root/memory_primitives runtime + gateway tests;
  /root/parallel_heads real model/media adapters; /root registry/CLI/API/install;
  /root/architecture independent boundary review. No detached/mock team runtime.
- Review corrected fixed-budget registry lookup, script HTTP error-class mismatch,
  native flag loss, duplicate MIME parameters and palette-alpha conversion.
  Implementation review approved; actual model/process-exit gate still pending.
- Windows runtime28 (one POSIX skip), gateway17, model11 (real CPU IR), CLI6,
  registry59 and installation closure passed. WSL filesystem/metadata9 passed,
  including FIFO and alias rejection. Native HTTP framing8 and gateway suite passed.
- Actual trained GPU camera and text gateway replays passed with authentication
  refusal checks. Actual GPU video export retained all three fixture frames and
  audio; VFR timestamps qualified separately. No microphone/camera opened.
- Speech checkpoint upgraded to pinned stateful export after actual beam_idx
  failure. Replay fixture conversion corrected after its transcript check failed.
  Windows GPU GenAI interpreter-exit/callback behavior is under investigation.
- TPU/AIPU production model adapters and physical NPU/power qualifications remain
  explicitly pending; prior head/vendor-image probes do not qualify these models.
- Final trained replay: correct full JFK on CPU/GPU; text answers/stateless request
  isolation and trained 4x SR parity passed. Actual CPU speech gateway exits cleanly.
  Windows GenAI2026.4.1 speech GPU fails interpreter teardown despite correct text;
  explicit request now refuses and AUTO reports restriction/selects CPU. Text/SR
  GPU stays enabled. SR f16 hint is optional with fixture-specific error bounds.
- Final new runtime test inventory:28 runtime,12 model/media,17 gateway,6 CLI.
  Native gateway425, registry59, HTTP framing8 and package closure passed. Optional
  PNG/video tests skip cleanly in the dependency-free native gate; full optional
  graph CI runs them. Public replay evidence excludes private SDK paths/capture.
- Combined final suite passed64 tests with one Windows POSIX FIFO skip (Linux case
  separately passed). Published functional commit6855039471a3fbbf2c4ad9f5e397d152bb29a04c.
  Graph CI38107182186 and portable-core CI38107182267 dispatched; online status
  pending at publication. No full-model TPU/AIPU qualification claim.

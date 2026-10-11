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
- Graph-runtime CI38107182186 completed successfully on both Linux and Windows at
  functional commit6855039, including optional model/media packages and CPU IR.
  Portable-core rerun38107182267 remains in progress; its unchanged core has the
  prior sanitizer/fuzz qualification and local native gateway/registry checks.

- Native throughput objective resumed with real Qwen3.5-0.8B and OLMoE checkpoints.
  New qwen36 CPU head bank reads shared committed hidden/known-token embedding;
  offsets2/3 run independently and target verifies/retains matching states.
- CPU output projection groups exact F32/F16/INT8 rows; MoE staging groups by
  distinct experts and preserves balanced holds. Strict and ASan/UBSan checks pass.
- First learned hidden-only head retained heldout IDs/LP but paired median0.975:
  no acceleration claim. Initial augmented training diverged; RMS normalization
  folded into weights plus seeded shuffled/clipped optimization stabilized it.
  Two-head heldout top1 .129/.0797 and bothprefix .0323 remain overfit, timing pending.
- Active-bank real-target gates prove rejection, acceptance, budget clipping,
  pin/continuation, sampling fallback, EOS/cancellation rollback. Synthetic oracle
  fixtures are correctness-only. Loader FIFO blocking and bank export overwrite
  fixed; native identity binds IDOT/INT8/tap and actual tensor/tokenizer bytes.
- New capture/paired benchmark tools, explicit qualified CLI flags and focused
  native CI added. All-family/device speed objective remains in progress.
- Requested regular-Colibri comparison completed: lookup enabled in both arms,
  same native Qwen3.5-0.8B INT8/integer-dot,8threads,3heldoutprompts*5pairs.
  Median arm decode39.829vs39.471tok/s;medianpairedratio1.01455,8wins7losses.
  IDs/logprobs/outputs identical. Results mixed:no clear repeatable speedup.
- Strata reference61b3fb5 verified clean: full default runtime CUDA/HIP-only;
  experimentalSYCL lacks supported localIntelUHD setup and cannot our0.8Bgeometry.
  OfficialRTX5070Q2_0FlashNext93.0at4K/73.7at128K retained as separate published
  references. No fabricated localStrata score or cross-hardware speed ratio.
- Final native focusedgate passed4C+32Python,training10+coverage1,converter6,
  generateddependencies8 and finalloader/kernelASan+UBSan. IndependentCPUreview
  approved. Tested native adapter and curated comparison ready for publication;
  overall all-family/device/throughput objective remains incomplete.

- User clarified that parallel future-token drafting remains the requirement;
  requested source is robertkarl/UnbiasedStrata. Its sequential MTP is excluded.
  Two CPU ports now share prepared activations/projection dispatch and INT8
  weight-block/sign work across verification rows inside Colibri.
- Real isolated15-pair studies: allports1.0129 median paired decode ratio,
  row1.0047,group1.0028; existing rank16 bank0.9779 (slower). All12 separate
  numeric output checks matched. Curated public evidence retains hashes and
  all measured pairs. OptionalwideS1 remains defaultoff; no model speed claim.
- Independent residual-head prototype preserves native raw-logit anchor and
  adds trained horizon-specific low-rank deltas, without head-to-head inputs.
  INT8sourceheldout69/279 and20/276,both10/279,meanprefix .28315; F16factor
  scores agree within1.34e-5. CSNHv2 explicitly distinguishes residualbanks;
  v1 APIs refuse them. Root caches raw logits before sampling and keeps them
  paired with accepted hiddenfrontier. Focused C and sanitizer gates pass.
  Live-target residualfrontier gates and freshprompttiming still running.
- Final CSNHv2 scorer/exporter/cache review approved;13interoptests and six
  native captured reference replays passed in scalar/parallel/sanitizer modes.
  Live residual first/mid rejection,fullaccept/newproposal,budget,pin,sampling,
  EOS,cancellation/postcancelcontinuation and precision refusals all passed.
  Combined native gate8C+62Python; dependency8 andVKlink2 passed. macOSmkdtemp
  feature declaration fixed; prior CPUfixtures now explicitly undefineVulkan.
- Fresh three-prompt15-pair studies complete with model/binary/bank hashes
  unchanged and exact numeric correctness. CPUports pairedmedian1.0331;
  residualheads isolated0.9706 (2.9%slower). One timed repeat perprompt yields
  gate-selected offset2accepted3/31,offset3neverjudged. This is a smallselected
  sample, not fulltraceaccuracy. Public evidence retains allpairs/counters;
  heads remain opt-in. Overall all-family/device throughput goal incomplete.
- User explicitly relaxed independent-parallel drafting: sequential Strata-style
  MTP now allowed. Used all15 actual pretrained Qwen3.5-0.8B MTP tensors with
  externalimmutable sidecar; defaulttarget conversion unchanged. Nativecell
  uses embeddingfirst,1+rawnorms,gatedGQA/SwiGLU,normalizedchainfeedback,
  sharedtargetreadout andKV-onlyknownpair prefill/catchup. Targetacceptedhidden
  replacesprovisionalMTPKV evenwhenallguessedtokensmatch. Firstdraftfeecharged,
  gatecalibratesplainT1; all sampled proposals bypassed safely.
- Converter16; combinednative10C+74Python; strict/scalar/ASan/UBSan kernel and
  frontier8/32modes passed. Realweightoraclemaxerror F32 1.55e-5/native8 4.77e-6.
  Actualrealmodeldepth1/2/3 targetIDs/LP/hidden identical; realcellforcedfirst,
  middle,cycle,fullaccept,budget,pin,prefix,EOS,cancel/postcancel,sampling and
  precision refusals passed. Dependency8 andVKlink2 passed; reviewapproved.
- MeasuredMTPalone:15 alternating pairs,3testrequests,64token cap,8CPUthreads,
  sameoptimizedbinaryoff/on; pairedmedian1.1313, all15wins,41.241→46.657 arm
  medians. CompleteforkvsunchangedColibri: pairedmedian1.1508, all15wins,
  40.827→45.997 arm medians. Bothnumericparity andasset/binaryhash stability
  gatespassed; no forced guesses/traces during timing. Overall all-family/device
  qualification remainspending; current denseCPU-MTPslice achieveditsgain.

- User approved deeper sequential MTP and larger MoE qualification. Added actual
  chosen-depth/source-acceptance evidence and same-binary maximum1 versus2/3
  studies. Thirty timed pairs and six numeric checks passed; paired medians
  1.0125547 and1.0009079, with graph-search regressions. Keep maximum1.
- Existing OLMoE PILOT0/1 measured at fixed cache8/16; thirty more timed pairs
  and six numeric checks passed, paired medians .9861831 and1.0062937. Increased
  cache hits did not justify enabling PILOT by default. Request-owned queue
  drain charges unused reads; late PROF handling and real worker-load evidence
  prevent false acceleration claims. Staged native-filesystem assets matched
  source hashes and were removed safely after measurement; originals retained.
- Final focused native gate and legacy dense-batch/OLMoE serve checks passed;
  47 benchmark Python tests; new cache/drain and legacy dense-batch ASan/UBSan
  passed. CI test setup fixes isolate old grouped-projection counters and restore
  Darwin feature declarations. Remote platform rerun pending. Direct review
  approved; all-family/device/precision qualification remains incomplete.

- User directed adoption of Strata-style drafting and reaffirmed the goal as
  faster usable Colibri through actual UnbiasedStrata runtime ports. Changed
  native/launcher/collector maximum to3; preserved explicit depth overrides,
  existing cost gate, exact target verification and historical measurements.
  Launcher5 and collector28 tests passed; native frontier passed; rebuilt real
  model selected maximum3 and matched explicit maximum1 IDs/LP/output for eight
  generated tokens. Direct review approved. This preference change does not
  complete the broader runtime/family/device goal.

- Native graph port: production Python inference replaced by persistent C17 process for text, Whisper recognition/English translation and trained SISR4x. Native C media prep, manifest/hash admission, framed IPC, timeout/cancel/retained cleanup ownership; pinned native SDK C-wrapper extensions preserve exact templates and Whisper bounds. Six real CPU/GPU HTTP replays run with all Python ML imports blocked; text token count/output and three CPU image fixtures match the offline reference exactly. Native GPU video preserves all3 trained frames, audio payload and timestamps. Windows/Linux C contracts and transport20/PNG7 pass; PNG ASan/UBSan, dependency8 and existing runtime/gateway/media/packaging checks pass. Decoder padding and out-of-palette index issues fixed. Independent backend/integration review approved. Ready for publication; broader native Strata/family/device goal remains incomplete.
- Final Windows entry qualification also fixes wide argv/environment-to-UTF8 conversion and vendor C-wrapper filesystem path decoding. Text, speech and trained SR succeed from Unicode model directories; core/GenAI DLL environment paths with Unicode work. Fresh native dependency preserves153 original exports and the same3 added C APIs. Final six production routes rerun with the UTF8-safe binary/wrapper and blocked Python ML imports.
- Published native C graph port and qualified evidence to origin/main at dc7d475886cb577916286faf55ca92b7e64a8946. Native graph slice complete; original all-family native throughput/accelerator goal remains active.

- User requested wrap-up and clarified their SSD/CPU/GPU/NPU/AIPU software stack is Modular Mojo. Recorded native C/MAX-or-Mojo-ABI integration target and honest boundary: published graph backend remains OpenVINO. Current repository/checked environments expose no Mojo/MAX installation; existing project/SDK location requested. Official MAX C API, Mojo native compilation and MAX26.6 generic accelerator/plugin routing verified from primary docs. Broader native throughput state handed off incomplete with no active workers; no untested backend or acceleration claim added. Documentation-only closure, no inference code changed.

- User prioritizes adaptable SSD/CPU/GPU/NPU/TPU/AIPU inference with state and bandwidth central, preserving Strata practices and adding other qualified inference techniques. Added backend-neutral native C per-work-unit mapping: exact workload/numeric/dtype qualification, state ownership/fences, directed measured transfer costs including staging/output, memory and latency budgets, conservative measured prefetch credit and incumbent hysteresis. Tests72 use synthetic profiles and establish policy behavior, not physical acceleration. Existing AUTO selection and model execution remain unchanged; actual Mojo SDK/profile-provider integration remains a separate required next slice. Brownfield map and explicit plan recorded.
- Published adaptive native policy at 01afb5c2306dc3806cfbb45852f6dfbd6189af05: final78 Windows/Linux/ASan/UBSan/fast-math checks and dependency8 passed. Policy slice complete; actual execution wiring and broader hardware/workload acceleration remain incomplete.

- Added actual native C Mojo/MAX backend with CLI probe/run, compatible MEF graph execution and external safetensor weights. Official MAX26.6.0/Mojo1.1.0 isolated SDK installed; native C CPU execution validated with5 exact synthetic fixtures and45 realSDK checks plus ASan/UBSan/leaks, including external weight lifetime. Signed4 storage expandsI8; I8/I16 inputs and I64 fixture math explicit. Runtime-visible accelerator_count0 and exact accelerator request refused without CPU fallback. Read/load/run/copy/publication/cleanup timings charged separately. Local qualification is WSL CPU, not bare-metal OS or otherdevice speed. Existing raw checkpoints/OpenVINO workloads remain separate; full state/SSD scheduling integration pending. Independent native C ABI review in progress.

- Independent Mojo C ABI review approved. Fixed the exact 128 MiB admission boundary; allocation-free exact/one-over checks cover all nine dtypes. Final 65 actual SDK checks, five exact native CPU graph replays, external-weight CLI ASan/UBSan/leaks, SDK-free launcher4 and Linux install closure2 pass. Complete input/model/result cleanup is charged before timing publication. Ready to publish this compatible-graph backend; broad workload/device/state/SSD integration remains incomplete.

- Published native Mojo/MAX backend to origin/main at ad468a584dde06b281fb918c589e1d03b4e630d4. Backend slice complete and independent review approved. Qualification is WSL CPU with zero accelerators; operator native bare-metal tests and original all-family/device acceleration remain incomplete. Source-qualified CI enabled; remote result pending.

- Exact implementation-commit native Mojo GitHub Linux CPU CI succeeded (run38159354356): official pinned SDK install, native C build, five exact fixtures including external weights, unavailable-accelerator refusal, 65 actual SDK ASan/UBSan/leak checks, installed closure and native probe. CI evidence retained and linked. Linux CPU CI does not qualify physical GPU/NPU/TPU/AIPU or full Colibri throughput.

- Investigated FreeToken distribution at pinned official commit 74a387777708b9650a027494bcb37d2ab9cf2443. Traced tensor-parallel rank/size and collectives separately from bandwidth-adaptive concurrent expert execution. Current source prefers contended CPU-MoE/PCIe-gather measurements, Q16 balanced integer splitting, per-GPU/format profiles and explicit partial-result merge. Recorded the single-target limitation of our mapper and the required partition/execute/join extension, preserving C state ownership and Mojo dispatch across native runtimes. Source/doc review only; no FreeToken installation, benchmark or execution port claimed.

- Added actual compiled Mojo common split decision ABI, optional native loader and shared Vulkan cold-decode split/retention/batch-recency LRU. C retains state, pins, backend calls and exact rank-order joins. Separate decode samples include staged setup, CPU/device work and fixed shared CPU work; extra/second-device/duplicate cohorts excluded. Serial cold uploads are charged explicitly, preserving existing double-buffered prefill and resident overlap. Grouped OLMoE CPU prefill holds each expert once, exact private rows/counts preserved. Independent review approved; 31,451 strict C /7,451 fast-math cases, 20,111 C/Mojo bit-exact comparisons, 63,238 actual Mojo/loader checks, sanitizers, native Vulkan references/retention/LRU/phase/failure gate and packaging/dependency checks pass. Nine frozen real OLMoE INT8 pairs on identical temporary native-filesystem assets: paired median0.9932801, three wins, exact IDs/text/counts and observed LPerror0. Software Vulkan only, no physical GPU speed gain; retain opt-in default. Earlier interrupted/changed-source report remains failed evidence and is not published as timing proof.
- Final full native Mojo/Vulkan ASan/UBSan/leak gate exposed an existing 975296-byte/968-allocation shutdown leak: normalization pipeline/shader/layouts, its previously local descriptor pool and qprep pool were not released. Retained pool ownership and added releases after device-idle; full sanitizer gate passes without suppressions. Rebuilt optimized tier/OLMoE/Qwen engines and reran native tier plus exact mixed-prefill gates, all pass. Independent cleanup review approved; published benchmark keeps its pre-cleanup hashes because the repair changes shutdown ownership only. Added the full leak gate to CI.

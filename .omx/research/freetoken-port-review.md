# FreeToken mechanism port: focused architecture review

Date: 2026-10-11. Owner/reviewer: `/root/architecture`. Production source read-only; no model run, compiler job or benchmark performed. Native in-session parallelism follows the team skill's small-session route. Verdict: **approved** for the implemented shared Vulkan cold-decode/native Mojo slice at the source identities recorded below. The initial architecture map and separate legacy findings remain historical context.

## Existing mechanisms, not new additions

| Existing source | Actual behavior |
|---|---|
| `hybrid_split.h:18-33` | DSV4 miss-upload count from legacy ratio, rounded to nearest integer; shared through `deepseek_v4_hybrid.h`. K3 adaptive inline fill budget is also already in this header. |
| `deepseek_v4.c:5610-5663` | Optional `DSV4_HYBRID` finds VRAM misses, selects upload count, enqueues drainable DMA and leaves a CPU subset. Views remain held until final cleanup. |
| `deepseek_v4.c:5731-5768` | CPU expert math runs while upload DMA is in flight, followed by drain. GPU group compute starts later at`:5781-5786`; existing behavior overlaps DMA with CPU, not resident GPU compute with CPU. |
| `vk_tier.h:3-29`, `vk_tier.c:328-428` | Shared hot-expert residency, history warm start, LFRU promotion/eviction, background uploader, budgets, optional second device and RAM/VRAM exclusivity already exist. |
| `olmoe.c:1491-1507`; `qwen36.c:3580-3608` | Submit resident GPU batch, compute CPU misses/shared work, join, recover device failures on CPU and merge in routing rank order. |
| `vk_tier.c:650-677`, `:866-1092` | Prefill cold-expert staging, two scratch halves and resident/streamed sub-batches already exist. Streaming defaults to multi-row GEMM steps; it is not decode S1 cold-miss dispatch. |
| `vk_tier.c:1342-1349` | Existing resident-work balancing changes GPU share when it keeps CPU waiting, using the optional in-RAM callback. |

New work should be described as a measured contended-cost miss split, safe integer policy and actual native Mojo dispatch coordination. Existing caches, asynchronous transfers, double buffering and resident CPU/GPU execution must not be credited as new mechanisms.

## Least-risk concrete path

First attach the common decision to the shared Vulkan tier and exercise it through OLMoE. Its engine-local `moe_vk_run` has simple private CPU output rows and a fixed rank-order merge. Qwen inherits the shared-tier change but adds a shared-expert buffer, shared format variants and the all-device sum optimization; extend coverage there after the same decision path is proved. There is no `moe_vk.h`; these wrappers are engine-local functions.

For a policy-only prefill slice, `vk_tier.c:885-886` is the cold/resident classification point and `:679-688` holds the current upload-versus-CPU row rule. Replace the decision under an opt-in flag and preserve the existing submit/join contract. This changes placement decisions, rather than creating a new concurrent CPU/DMA execution path.

For actual concurrent decode-miss execution, start with S1 and equal-geometry unique experts. Existing `vkt_step_rows:690-692` / issue dispatch do not automatically stream such steps. A selected cold GPU subset requires real staging/dispatch orchestration, not just marking its routing mask. Cached GPU work should be submitted before the cold load/transfer pipeline, with CPU math running concurrently. The current big-step scheduler can accumulate residents without submitting them until after its first cold group fills (`:1006-1036`); small steps therefore do not acquire this overlap merely by enabling streaming.

Keep C responsible for model/KV state, cache handles, source pinning, output storage and deterministic merge. The native Mojo coordinator can choose backend/partition from measured costs and actual available-backend capability inputs; MAX/OpenVINO/Vulkan are executor implementations underneath that decision. Compile and call a real Mojo C-ABI export/shared artifact from the native engine. A Python decision or an unused policy module does not establish the requested common mapping/dispatch layer.

A minimal ABI can use versioned scalar/POD arguments: bounded miss count, available-backend bits, CPU-under-DMA cost, DMA-under-CPU cost, GPU-miss compute/launch cost, resident GPU cost, and a caller-owned partition result. Invalid/unmeasured inputs must preserve the historical path or explicitly refuse an enabled required coordinator; never claim that a C fallback executed Mojo. No Mojo-owned model state or private backend object layout is needed. A Linux Mojo artifact cannot silently serve as a Windows native DLL.

## Policy and correctness traps

1. **Unbounded floating-to-int conversion.** `hybrid_split.h:29-32` casts before clamping. NaN, infinity and huge finite ratios can invoke undefined conversion; the EMA at`:39-42` also admits NaN. New policies need finite validation and saturation before any cast. Invalid input behavior and old feature-off behavior should remain explicit.
2. **Contended rate interpretation.** The legacy derivation assumes uncontended host bandwidth `Bh`, with CPU rate `Bh-Bp`. DSV4 samples host execution while DMA contends (`deepseek_v4.c:5731-5749`), so that measured rate is already `Bc`. For equal-byte, compute-excluded branches, balancing directly measured rates gives `q=m*Bp/(Bp+Bc)`, rather than `m*Bp/Bc`. GPU compute/launch/resident fees must be included in actual end-to-end branch costs.
3. **Nearest rounding is not integer minimax.** Evaluate floor/ceil and endpoints using predicted branch makespans. Example: m6, Bp1, uncontended Bh4 implies Bc3 and a continuous q1.5. q1 finishes in5/3 units; q2 takes2 units. Nearest rounding chooses the slower split. Optional fixed/empty-branch fees strengthen the need for cost evaluation.
4. **Existing DSV4 duplicate fallback on allocation failure.** `gpu_ok` starts as allocation success at`:5720-5726`. If false, CPU loop`:5733-5741` already computes all experts, then fallback`:5791-5801` computes GPU-resident experts again. An explicit completed/CPU mask or scheduled-state flag must ensure each contribution is added once. Ordinary launch failure after a successful allocation is a different state.
5. **Drain success/source lifetime.** The drain result at`:5758` is ignored. New dispatch must not commit residency, compute uploaded weights or release DMA source slabs before successful completion; failed/partial work needs a defined drain-and-fallback path. Enqueue timing is not transfer bandwidth.
6. **Engine-thread-only tier calls.** OLMoE CPU helper`:1470-1473` and Qwen helper`:3563` call `vkt_note`; tier callbacks/bookkeeping require the engine thread (`vk_tier.h:181`,`:190`). Moving these helpers wholesale to a worker while issue mutates tier state introduces races. Worker work should be pure expert math over pinned slots/private outputs; defer promotion and bookkeeping to the engine after joins.
7. **Measurement scope.** DSV4 fill window includes the CPU interval and samples only drains that waited. Vulkan issue-to-join CPU time includes engine/shared work. Do not label those intervals pure PCIe or DRAM bandwidth. Record which costs were sampled under contention, hardware/format/model geometry and which controller determined each split. Disable or account for the old resident-share controller during policy ablation.

## Join/residency safety contracts

- Every valid `(row, rank)` must end on exactly one completed CPU or GPU output, including partial load, allocation failure, submit failure and device loss. Invalid rank slots remain unused; duplicate expert IDs across rows still map back to each original row/rank.
- Pin source slot bytes through conversion/transfer completion. Do not let PILOT/eviction recycle a busy slot or publish a stale ticket as resident. Preserve existing RAM/VRAM/SSD budgets, histories and formats.
- Preserve `vkt_issue`/`taken` semantics: only accepted GPU submissions set ownership bits. Always drain/join accepted work before returning or taking another layer/step; no dense Vulkan calls from parallel regions during an in-flight expert batch (`vk_tier.h:95-99`).
- On a failed join, recompute only skipped GPU assignments and clear their ownership bits, as OLMoE`:1496-1498` and Qwen`:3586-3589` already do. Merge CPU/device results once in original routing rank order; preserve activation/weighting and V4 bf16/E4M3 semantics.
- All-device sum optimization may run only when every valid routed assignment actually completed on that device (`vk_tier.c:1371-1388`). A policy-selected mask is not completion proof.
- Reject concurrent/reentrant dispatch or stale completion; teardown, cancellation and allocation errors must retain ownership until outstanding work is drained.

## Verification hooks and feasibility

Policy tests: finite/nonfinite rates, zero/negative/missing samples, extreme finite saturation, integer brute-force oracle, fixed resident/launch fees, heterogeneous byte sizes if supported, endpoint splits, feature-off identity and deterministic ties. Dispatcher tests need deterministic barriers proving CPU and transfer/GPU work actually overlap, plus each failure stage and exactly-once outputs; sleep-based timing alone is not proof.

Existing `tests/test_vk_tier.c` is the native shader/reference gate: every accepted source format, mixed rank-order sums, warm/history order, moving-hot-set eviction, partial rows, promotion while in-flight, accounting/budgets, V4 activation, extra layers and cold streaming. It already uses an independently decoded CPU reference and a2e-3 tolerance. Command: `make -C c vk-tier-check VK=1` with a selected ICD. Lavapipe gives functional Vulkan evidence, not physical GPU performance. Add the new cold split, concurrency and failure cases to this existing path rather than inventing a separate graph demo.

Strict engine verification should compare target IDs and numeric output/logprobs for deterministic native decoding under feature-off/on, pin/prefix reuse and cancellation. GPU/CPU arithmetic may differ; numerical tolerance and near-tie handling must be declared. Use a real compatible MoE checkpoint for model-speed claims. The existing Qwen3.5-0.8B dense checkpoint cannot exercise routed expert cache misses. Synthetic expert fixtures establish execution/joins only.

Light tool inventory: Windows has `C:/VulkanSDK/1.4.350.0/Bin/glslc.exe` and `C:/WINDOWS/system32/vulkaninfo.exe`. Kali WSL has multiple ICD manifests including Lavapipe/Intel/gfxstream, but no `/usr/include/vulkan/vulkan.h`, `/usr/bin/glslc` or `/usr/bin/vulkaninfo` at review time. Installing native dev/tools is needed for the WSL Mojo+Vulkan test path. ICD manifests and the earlier Intel OpenVINO replay do not establish actual Vulkan execution or contended bandwidth. No hardware probe, driver change, model download or heavy benchmark was performed here.

Parent owns implementation and its native Mojo/Vulkan/engine evidence. The implementation review below closes the scoped source review; this architecture map does not qualify a physical hardware speed claim.

## Implemented shared-tier slice review

The implementation diff is reviewed. Verdict: **approved** for this shared-tier native Mojo/Vulkan slice. No remaining concrete new implementation blocker was found in the final recorded source. The final decode-to-prefill transition and source-retention fixes are included; broader backend/device qualification is explicitly separate.

Actual behavior is deliberately narrower than the first architecture proposal: cold decode uploads are staged serially before CPU/device computation; the ABI uses `transfer_overlap=0` and charges that serial cost. Existing Vulkan GPU/CPU compute then overlaps. This slice does not introduce or prove concurrent cold DMA/CPU work. The old DSV4 hybrid concerns above remain separate legacy-path findings; this implementation uses the shared Vulkan taken/join contract.

The native Mojo library is a real common planner C ABI, loaded and called by the shared tier. It chooses integer partitions; C still owns model/cache state, upload buffers, native backends and fences. The controlled native/Mojo tier tests execute actual Vulkan shaders on Lavapipe. Other MAX/OpenVINO/common-layer caller wiring and physical GPU integration remain separate qualification slices.

### Findings corrected during implementation review

- **Phase isolation:** initial code reused legacy `cpu_row_ms` for live decode costs. It now has separate `split_cpu_row_ms`, `split_upload_ms`, `split_gpu_row_ms` and `split_host_base_ms`. Prefill metrics cannot seed the new decode plan.
- **Shared CPU work:** Qwen now calls `vkt_host_base_done` after its always-CPU shared expert work, charging that separate base in live costs. `vkt_host_done` records expert-stage completion even when CPU pair count is zero.
- **Exact caller profiles:** trusted nine-cost overrides now pass through exactly. Automatic resident-device base calculation applies only to live estimates.
- **Matching cohorts:** sampling eligibility is captured for a unique-expert, main-layer, single-device S1 workload. Extra-format MTP layers, second-device cohorts and duplicate expert assignments defer sampling/planning. Valid host assignment counts exclude unused negative IDs.
- **Capacity:** misses use successfully allocated staging capacity and backend scratch-fit row capacity after resident work, rather than requested staging-slot count. Calibration requires actual capacity and at least two misses. Only one cold expert is seeded once; missing/failed timing evidence does not trigger repeated bootstrap.
- **Telemetry:** cold-device/CPU counts are explicitly planned counts. Failed loads/submits can change the actual ownership mask; execution proof continues to come from taken/join, not the plan or predicted time.
- **Build/install:** policy/loader objects have separate tracked compilation, generated dependencies and cleanup. The test compiles one translation unit and links its tracked helper object. Install/archive include policy/loader headers and source, actual Mojo source/build tool and optional native planner artifact.
- **Recency and retained cold fills:** replacement uses an independent per-batch recency clock; all active IDs tie/protect one another before a cold expert is offered for promotion. The engine-thread promotion call holds each source handle through the copy into the bounded upload queue. Source release and existing slot/fence ownership remain intact.
- **Decode-to-prefill transition:** an accepted `vkt_stream_prefetch` establishes a non-decode phase before staging. Refused prefetch calls preserve an active cohort. The regression compares decode samples before/after prefill staging and confirms they are unchanged.

### Native ABI, cleanup and join audit

`runtime_split.h/.c` and the compiled `mojo_split_dispatch.mojo` share a versioned flat ABI. Costs are validated as finite nonnegative binary64 values without floating-point casts; exact software addition/count multiplication handles subnormals, overflow and lower-transfer ties even under FTZ/fast-math. Candidate search is bounded by1024 misses/cap, evaluates valid integer makespans, and publishes both count/time outputs only after a successful decision. Output overlap is rejected; input data is snapshotted before output writes.

The native loader checks version/init/plan exports, initializes the genuine Mojo runtime, keeps the shared-library handle live for all borrowed planner calls and closes failed opens. Requested absent/incompatible Mojo refuses rather than silently invoking the C oracle. The planner call is synchronous and no uploader/backend worker retains its function pointer; serialized shutdown may close the planner before GPU fences safely because subsequent cleanup invokes no planner. Vulkan buffers/expert slots are still fenced before their original teardown.

Cold selection changes expert placement, not IDs/weights. Staging load handles and source bytes retain their existing release rules. Ownership bits are set only by successful native submits. Unsubmitted/failed cold work stays CPU; failed joins cause the engine to recompute the previously skipped GPU assignments, clear taken bits and merge each routed contribution once in rank order. The all-device sum path still requires every valid assignment to complete on-device. New host measurement hooks remain on the engine thread and do not move `vkt_note` into a concurrent worker.

The replacement option now uses its own per-expert-batch recency clock/stamp, preserving default LFRU. Active IDs in the same batch deliberately tie/protect one another; it is batch-granularity recency rather than strict per-access ordering. Newly GPU-staged decode experts can be offered for persistent retention while load handles remain held; offers occur on the engine thread after parallel staging and before release. Existing bounded uploader/fences own the published residency.

### Retention transition finding resolved

The initial retention addition allowed `vkt_stream_prefetch(layer,S>=gemm_rows)` to inherit `T.split_decode` from a completed S1 decode. That could update decode upload samples and promote prefill experts under the new retention hook. The final code clears the phase after its refusal guard and before allocation/fill. The final test invokes prefetch after decode calibration and checks the complete sample vector remains unchanged. The guard order also preserves a genuinely active cohort when prefetch is refused. This finding is resolved.

The final OLMoE change also groups prefill CPU assignments by expert: it acquires an expert once, computes all assigned rows into the same private row/rank outputs, records held reuse, offers residency while held and then releases. The final rank-order merge is unchanged. This source-cache reuse is separate from decode split policy and from new transfer overlap.

### Evidence and qualification limits

Read `outputs-native/freetoken-final-vulkan.log` from the final owner campaign: it ends `PASS`, including existing format/streaming/fault/mixed cases, cold native and actual compiled Mojo cases, retained cold fills and LRU rotation. The controlled profiles exercise CPU-only/mixed/all-device ownership; their numbers are not measured hardware profiles. The new transition regression is present in that tested source. This reviewer independently ran the four builder mock tests successfully and performed source/ABI/lifetime review; no additional compiler, model or heavy benchmark work was performed during the owner campaign.

The earlier `outputs-native/freetoken-olmoe-live-paired.json` inspected during review has failed qualification (`benchmark input assets changed`) and is not timing evidence. The later frozen rerun is recorded in `experimental/strata/results/freetoken-olmoe-mojo-software.json`, SHA256 `9b32e606a51cbc5fd1fe52ab79292f1d0c9d88573cff0f3c2806779813b153f5`. This reviewer independently checked its `complete` status, nine paired records, byte-equivalent before/after identity records and exact IDs/text/counts across the three correctness prompts. The observed maximum absolute logprob difference is zero at the reported protocol precision. The metric covers prefill, decode and request-owned drain; it is not isolated decode throughput.

The nine-pair median whole-request speed ratio is `0.9932801343973122`, approximately 0.7% slower, with three candidate wins. The native Mojo planner is invoked in the shared-tier path; on llvmpipe software Vulkan it chooses cold CPU work after the one calibration. This run demonstrates a real native MoE workload and correct dispatch with no gain on this setup. The feature remains opt-in. Hashing warmed the page cache before measurement; this is not a cold-storage benchmark or physical GPU speed evidence.

The benchmark predates the cleanup-only repair below: its `backend_vulkan.c` hash is `dc79dda7628b764f6e29aa6119c31ab4be5e65a1f9f1ec234977d0f2f51dadc0`, and its native binary hash is `8f2f64819948d36c27e00dbd814107c73a1b95d6d133d47ff2d0c0190eb545d5`. The recorded post-repair source is separately reviewed; the model comparison is not falsely attributed to that rebuilt binary.

### Cleanup-only Vulkan repair

The final LeakSanitizer campaign found existing normalization/qprep objects omitted by shutdown. The repair retains the normalization descriptor pool in `G.dpool_nrm` rather than a local handle and destroys `qprep_pool`, `dpool_nrm`, `pipe_nrm`, `plyt_nrm`, `dsl_nrm` and `shader_nrm`. Shutdown first waits for the device, then frees command/descriptors/pipelines/layouts, and finally destroys the device and zeroes `G`. Each added handle is released once and no inference math or dispatch branch changes. Source review: **approved**.

Owner's final actual compiled Mojo/full-tier ASan+UBSan rerun returned exit0 with leak detection enabled and no suppressions. Independently read `outputs-native/freetoken-vulkan-asan.log`: `PASS` is present, with no `ERROR`, sanitizer summary, runtime-error or LeakSanitizer diagnostic. Log SHA256 is `9b2336cca702d69588c15b2d7a4f174f5dc12eeb1e3c543408d96d30b62b8eaa`. This validates the post-cleanup tier gate; the model benchmark remains attached to its own pre-cleanup binary/source identities above. The workflow now triggers on backend changes and repeats the full Vulkan sanitizer gate with `detect_leaks=1:halt_on_error=1`.

The implementation is opt-in and its ordinary LFRU/prefill paths remain separate. Lavapipe is software Vulkan functional evidence. Physical GPU bandwidth, actual new DMA overlap, trained-MoE speedup, native Windows Mojo dispatch, broader engine/device/profile variants and all-backend common-layer wiring are not qualified by this source review and shader gate. Legacy DSV4 ratio/fallback/drain findings remain outside this approved shared-tier slice.

Approved source identities (later owner changes require an updated review recording):

- `c/vk_tier.c`: `7f2707821d61aa7ab83bf66977b8a8c1f74f15949414cf412de964f52034d5a0`
- `c/backend_vulkan.c` (post-cleanup): `c2f90bfd8329203a045529c677621793c9344915aa2314418165d132e04734c5`
- `c/vk_tier.h`: `f04e04df92add6a83cef619f261c8256aa688f8a3991a7af7e6ee6359177646c`
- `c/runtime_split.c`: `e7a941e7aad02a461fc3ab85be6d72671438ae57c5b34975675c2e7155c03ba4`
- `c/runtime_split.h`: `69de6d0e091147e11f10cae4af04deee73bdbd7ff24a0d0be81f97e36fe06f76`
- `c/mojo_split_dispatch.mojo`: `142f1d7efed65eb9dd6c5e74eafb42c19e944b3dc254a08e01262de0f7fd9f2f`
- `c/mojo_split_loader.c`: `07ddfb3899863f62ae9580f4ad79623fa7b3e5f99cf1a970fc3a6bc083207269`
- `c/mojo_split_loader.h`: `b9d3102ebdd1f62282212cccfcf597e6a2cb7b6a2aefb7944d97b7139e4b7478`
- `c/olmoe.c`: `77c5dbffafe007a196f570ff56479a9b6d9547f43da514209751b228e396af97`
- `c/qwen36.c`: `0505b5789bd0a9b8b755594cec0bc7d7017dacb2d105f9fc39d1945199ae131d`
- `c/Makefile`: `976a0ebf9fdd9fa01b2627a8bd08bc2ca68f63ee5794233cf9eda62f0374017c`
- `c/tests/test_vk_tier.c`: `0b3c2ea40400e9b9e16fa0aaa55e575d06238c28d9dbc1127879dc1d81350c5c`
- `c/tools/build_mojo_dispatch.py`: `94de736e8be71f64f31c52bc392ea103008f8eca04e227a981a44444cac789c8`

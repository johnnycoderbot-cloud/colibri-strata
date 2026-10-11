# Native concurrent expert execution with Mojo

The fork now has a compiled Mojo mapping/dispatch decision layer connected to
the shared native Vulkan expert tier. Colibri's C engines continue to own model
state, caches, weight handles, backend calls and output merges. This is actual
native Mojo code called through a C ABI, without Python model inference.

The new decoder policy partitions unique cold experts between CPU computation
and staging/execution on the device. It minimizes the complete estimated
critical path over bounded integer partitions: serial setup plus the slower
compute branch plus join. Queue delays, fixed CPU/device work and transfer costs
are explicit. GPU-resident experts remain in fixed device work. Ties transfer
fewer experts; absent capability is not zero-cost execution.

Mojo and C implement the same flat, versioned ABI in runtime_split.h. The
compiled implementations are checked against each other bit for bit. Inputs,
NaN/Inf, overflow, subnormals, aliases and unavailable branches are tested.
An explicitly requested missing/incompatible Mojo library fails rather than
quietly using another planner.

## What changed in execution

- One-row main-layer decode can now stage a measured subset of cold experts,
  then run the device batch alongside the remaining CPU work.
- Successful cold fills can warm the persistent cache through owned copies and
  the existing bounded uploader. Publication and eviction respect device fences.
- Optional batch-recency LRU follows actual routing, protecting every expert
  active in the current batch. Existing LFRU remains the default.
- Live decode samples are separate from prefill, extra MTP formats and
  second-device cohorts. One bounded calibration split obtains startup samples;
  missing/failed measurements keep cold work on CPU afterward.
- OLMoE mixed-path CPU prefill holds each expert for all its assigned rows,
  avoiding repeated cache reads. Private row positions and final rank-order
  accumulation are unchanged.

Colibri already supplied resident CPU/device overlap, two-device resident
dispatch and double-buffered prefill streaming. Those paths are retained.
The new cold Vulkan upload stage is serial before concurrent compute, and is
charged that way. This port does not claim new simultaneous CPU/DMA execution.

## Build and use

Use a supported native Linux/macOS Mojo SDK environment; MAX26.6.0 installs the
qualified Mojo1.1.0 compiler. Native Windows cannot load the WSL shared object.

~~~sh
.venv-mojo/bin/python c/tools/build_mojo_dispatch.py
make -C c olmoe VK=1
export COLI_VULKAN=1
export COLI_MOJO_DISPATCH="$PWD/c/libcoli_mojo_dispatch.so"
export COLI_VK_TIER_SPLIT=mojo
export COLI_VK_TIER_POLICY=lru
python3 c/coli run --model NATIVE_OLMOE -p "Explain a suspension bridge."
~~~

COLI_VK_TIER_SPLIT=off preserves the existing cold decode route.
COLI_VK_TIER_SPLIT=native uses the equivalent portable C planner for comparison.
The feature remains opt-in until useful speed gains are qualified on hardware.
Normal builds need no Mojo SDK; the native loader adds no SDK link dependency.
Rebuild the optional library on the destination host with its installed SDK.

The native interface vkt_split_profile accepts a trusted caller's exact full
cost profile for the current engine/device/one-row workload. Fixed base work
already includes its resident experts. Such a profile is not hardware execution
proof. The default live path derives conservative costs from actual decode
staging, host work, device issue/timestamps and exposed join/readback.

## Verification and benchmark

The native tier tests execute CPU-only, mixed and device-only cold decisions,
live calibration, retained/cached replay, LRU eviction and decode-to-prefill
transitions, alongside its existing format, mixed sum and failed-staging tests.
The controller's counters label planned work; taken/join results determine
actual execution and exactly-once output ownership.

The complete native Mojo/Vulkan gate also passes ASan, UBSan and leak checking.
That final check exposed and fixed missing Vulkan normalization pipeline,
shader and descriptor-pool releases at shutdown. CI runs the same full gate
with leak detection enabled.

~~~sh
make -C c runtime-split-check
make -C c vk-tier-check VK=1
python3 c/tools/native_moe_split_bench.py --engine c/olmoe \
  --model NATIVE_OLMOE --prompts three-prompts.json --output NEW_REPORT.json \
  --candidate mojo --mojo-library c/libcoli_mojo_dispatch.so \
  --tokens 8 --repeats 3
~~~

The collector compares split-off/on in the same binary, preserving model,
source, shader, plugin and prompt hashes. Every request checks exact token IDs,
counts and text plus a declared finite logprob-error bound. Whole-request DONE
timing includes prefill; it is not an isolated decode metric. Both arms use
SYNC=1 for promotion drain at quiescent joins. Hashing warms the file cache;
these runs do not establish cold SSD performance.

Local execution uses WSL and Lavapipe, a CPU software Vulkan implementation.
This establishes native execution/ownership correctness and a software-path
comparison; it does not establish physical GPU speed or power savings.
Physical GPU/NPU/TPU/AIPU and broader model/precision qualification remain
separate gates.

The frozen OLMoE-1B-7B native INT8 comparison completed nine alternating timed
pairs across three requests, with eight generated tokens per request. Exact
token IDs, text, counts and observed logprobs matched. The paired median
whole-request speed ratio was 0.99328 (about 0.7% slower), with three candidate
wins. This software-path measurement shows no repeatable speed gain; it is not
a physical GPU benchmark, and the splitter stays opt-in.
The report preserves the exact source and binary hashes measured. It predates
the cleanup-only Vulkan shutdown repair; final optimized and sanitizer gates
were rerun on that repair. No inference operation changed in the repair.

The common Mojo planner ABI can govern other compatible native execution
adapters. Its current production caller is the shared Vulkan expert tier.
MAX graph execution and OpenVINO workload routing are separate existing
adapters; they do not yet all pass through this coordinator.
Full network/tensor-parallel collectives and semantic-state anchor ports are
not introduced by this cold-expert slice.

# State and bandwidth aware inference

The objective is an adaptable native runtime for SSD/NVMe, RAM/CPU, GPU, NPU,
TPU and AIPU. It should carry useful practices from UnbiasedStrata and other
inference engines into text, speech, translation, image/video processing and
other model workloads. Colibri's core stays C; the user's Modular/Mojo stack is
the intended additional native backend.

The mapping unit is a workload stage with exact model/numerical identity and
measured cost. Devices are selected by the complete execution path, including
queueing, resident state, cold weights, input/output copies and staging. Verified shared-memory aliases can avoid copy costs while retaining
state fences and compatibility checks. A faster
kernel can lose after copying a large KV or recurrent state. State remains with
its current owner until a compatible destination and completed consumer fence
allow migration. SSD supplies storage and spill capacity; it is not a compute
device.

## Implemented first slice

The native C API in c/runtime_map.h and c/runtime_map.c ranks one work unit from
caller-provided measurements. It recognizes every tier above, prices directed
transfer paths through RAM/other staging tiers, enforces memory budgets, refuses
expired/unverified or numerically mismatched profiles, preserves pinned state,
enforces complete-cost latency limits, and applies an incumbent saving threshold to avoid churn. INT4/8/16 and floating
dtype requirements are explicit masks.

The estimated score is queue + compute + input transfers + output transfers,
less bounded measured overlap credit. Credit only applies to explicitly
prefetchable non-state inputs. Missing links never become an invented direct
SSD-to-device path. The source owner is unchanged; this function neither moves
buffers nor executes models. Work-unit budgets include that unit's resident
inputs; the caller accounts for shared pools, reservations and contention.

Compile adapters against the matching header. Identity handles must bind the
exact checkpoint, operation/phase/shape/batch and numerical/state encoding; they
are caller-issued identities, not security hashes. Profiles must measure effective
bandwidth and queue delay under relevant load. This is a bounded per-unit policy,
not a concurrent global resource allocator.

These 78 synthetic policy checks passed Windows MSVC, Linux GCC, ASan/UBSan and
fast-math builds. They establish policy behavior, not acceleration. The mapper
is not yet wired into the existing CLI AUTO selector or a Mojo device driver.

## Practices to port and qualify

| Practice | Purpose and required boundary |
|---|---|
| Strata-style sequential MTP and target verification | Amortize target passes while preserving accepted state, EOS, budgets and rollback. Existing qualified qwen36 CPU path remains available. |
| Double-buffered prefetch and persistent hot experts | Overlap eligible weight transfers with compute; measure misses and waits and retain consumer/transfer fences. |
| Chunked prefill and phase-specific placement | Use large arithmetic-friendly batches for prefill, then minimize state/weight movement during decode. |
| FreeToken-style concurrent expert execution | Split missing experts between compatible compute/transfer branches using achieved bandwidth under contention; explicitly join outputs and preserve state ownership. |
| Paged KV and prefix reuse | Reuse compatible immutable prompt state and account for growing cache memory; recurrent/SSM state needs its own layout. |
| Continuous batching and backpressure | Admit work within memory and latency budgets; keep conversation/state ownership isolated. |
| Fused kernels, prepared activations and graph capture | Reduce memory traffic and launch overhead for supported stable shapes. |
| Quantized weights/state | Reduce bytes moved; distinguish integer arithmetic from storage codecs and validate quality separately. |
| Encoder/feature reuse | Cache compatible text/image/audio encoder results with exact model and input identity. |

These are selected integration requirements, not claims that every practice is
already implemented. [FreeToken distribution investigation](../.omx/research/freetoken-distribution.md)
distinguishes its concurrent CPU/GPU expert split from tensor-parallel rank
metadata. Our current mapper chooses one target; partition/execute/join is a
separate required extension.
[vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
avoids repeated shared-prefix prefill; it does not by itself speed generation of
new tokens. [vLLM Ascend's layer/sparse offload design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.html)
illustrates why prefill and decode need different state-transfer strategies and
why physical buffers cannot be reused before all consumers finish. Its specific
sparse layouts are not universal adapters.
[NVIDIA Dynamo encoder caching](https://docs.nvidia.com/dynamo/dev/backends/tensor-rt-llm/tensorrt-llm-multimodal)
provides a useful multimodal reuse pattern. Mojo/MAX integration uses a real
[native C boundary](https://max.modular.com/api/c/) and the user's installed
device plugins, rather than translating device names into support claims.

For other inference workloads, keep the same policy but use their actual state:
audio feature/decoder state, video temporal state where supported, diffusion
latents/scheduler state, or encoder embeddings. Do not carry state across models
or frames that do not expose a compatible reuse contract.

## Next execution boundary

Connect the user's native SDK/plugin, attach measured stage/link profiles and
wire the mapper into a real workload. Executors must reserve resources, follow
the selected topology, snapshot/restore state and confirm successful ownership
handoff before committing a new owner. Revalidate the profile if topology or
capacity changes. Existing unmeasured routes retain their current policy.

Publish same-model comparisons charging all reads, copies, verification and
cleanup, with correctness and latency/throughput evidence. A portable policy,
dtype enum, compiled kernel or device discovery alone does not establish that
the entire workload is accelerated.

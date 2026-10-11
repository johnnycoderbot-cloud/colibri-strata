# FreeToken distribution and bandwidth-adaptive execution

User requested investigation of FreeToken's distribution feature after
clarifying that Mojo should map/dispatch native accelerator runtimes.
Read-only source inspection; FreeToken was not installed or benchmarked.
Pinned official source: FlashML-org/FreeToken at
74a387777708b9650a027494bcb37d2ab9cf2443.

## Two distinct mechanisms

FreeToken exposes tensor-parallel membership through
[DistributedInfo(rank, size)](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/python/freetoken/distributed/info.py).
The server accepts --tensor-parallel-size / --tp-size and GPU selections
corresponding to ranks. Its
[communicator](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/python/freetoken/distributed/impl.py)
provides all-reduce/all-gather through PyTorch distributed or PyNCCL.
Rank metadata alone does not schedule heterogeneous accelerators.

Its separate hybrid MoE mechanism partitions each layer's unique missing
experts. Resident experts run on the GPU; some misses are fetched into the GPU
cache, while the CPU computes the other misses from host banks. Partial outputs
merge. The
[cache implementation](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/python/freetoken/moe/offload_cache.py)
supports fixed fetch caps and a measured per-step fraction.

## Actual current split policy

[benchbw.py](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/python/freetoken/moe/benchbw.py)
measures real CPU expert kernels and PCIe expert gathers, separately and
concurrently. It recommends hybrid above a configurable CPU/PCIe bandwidth
threshold (default 2), with CPU-incompatible formats remaining offload-only.
Per-GPU profiles distinguish cards/slots and expert formats.

[load_hybrid_fetch_fraction](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/python/freetoken/moe/bench_profile.py)
prefers concurrently measured bandwidths:

    fraction_to_fetch = PCIe_overlap / (PCIe_overlap + CPU_overlap)

The legacy standalone fallback is min(1, PCIe / CPU), assuming DMA reduces
the bandwidth available for CPU work. Do not substitute advertised RAM speeds
or PCIe lane ceilings for these achieved measurements.

[offload_kernels.py](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/python/freetoken/moe/offload_kernels.py)
encodes the fraction in Q16 and chooses floor/ceil counts by minimizing the
slower branch's estimated time. It deduplicates experts, preserves hits and
marks unfetched misses for CPU work. Current integer selection can fetch zero;
the older paper's minimum-one-fill description is not a universal current rule.
The inspected
[tests](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/tests/moe/test_hybrid_fetch.py)
cover rounding, overlap-profile priority and CPU/GPU decision parity.
Those upstream tests were read, not run in this checkout.

## Residency, state and prefill

The [paper](https://arxiv.org/html/2608.16157v1) describes a shared LRU expert
cache, two full-layer prefill buffers, semantic-boundary recurrent-state
checkpoints and cache resizing at scheduler safe points. Its host expert pool
remains authoritative. These mechanisms address residency, transfer overlap
and reusable state alongside the compute split. They do not establish universal
NPU/TPU/AIPU support or a speedup for our fork.

## Concrete fit for Colibri

Our c/runtime_map.h/.c ranks one complete work unit and returns one target.
It is not a concurrent partitioner. The native MAX graph bridge likewise
selects one explicit device; it does not implement FreeToken co-execution.

The next implementation boundary is an explicit partition/execute/join contract
for independent experts or other genuinely separable work. Preserve the C core
and state ownership; use native Mojo bindings to dispatch qualified MAX,
OpenVINO, Vulkan or device-specific SDK adapters. This is our adaptation
proposal, not a capability demonstrated by FreeToken.

Required costs include contention, GPU compute after transfer, activation copies,
merge, queues, memory reservations and state fences. Measure by actual device,
operation/phase/shape/batch and precision. Extend beyond MoE only when the
workload exposes valid partitions and a compatible merge/state contract.
Keep nonseparable state with its owner unless a measured, fenced handoff wins.

Qualification must compare the same checkpoint and output contract against the
unchanged native path, charging cold SSD reads, transfers, compute and cleanup.
This investigation changes no executor and makes no throughput claim.

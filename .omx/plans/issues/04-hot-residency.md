Integrate decaying expert-popularity selection with Colibri's existing placement
and caches, including actual GPU admission/promotion/eviction event fencing.

Acceptance:

- Reserve dense weights, KV, workspace and staging buffers before hot-set selection.
- Preserve every target router selection and original expert weight precision.
- Fence in-flight consumers and promotion copies; cold experts retain CPU/demand fallback.
- Handle backend allocation failures and avoid thrashing and layer starvation.
- Show bounded peak VRAM, per-layer hit rates, promoted/evicted bytes and overhead
  on trace-driven and real model workloads.
- Describe policy reservations separately from actual VRAM residency/OS page locking.

See c/strata_memory.h, docs/strata-fork.md and docs/strata-benchmarks.md.

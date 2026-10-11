Bind the portable two-slot prefetch lifecycle to actual CUDA/HIP expert transfers
and Colibri's existing expert-store interfaces. Implement budgeted chunked prefill
without importing Qwen-only geometry into other model families.

Acceptance:

- Separate copy/compute streams; no reuse until transfer and consumer events end.
- Predicted routing is advisory; demand misses retain the synchronous exact path.
- Allocation/device loss and cancellation drain safely with no use-after-free.
- Sweep fitting chunk sizes from 512 to 8192, respecting per-device workspace.
- Preserve routed experts and numeric kernels, record demand/prefetch/wasted bytes,
  peak VRAM and real GPU event overlap rather than claiming eliminated PCIe cost.
- Report TTFT/prefill and decode changes across cold/warm/context workloads.

See c/strata_memory.h and docs/strata-roadmap.md. Fake-event tests validate the
interface; real asynchronous DMA overlap has not yet been demonstrated.

# Adaptable native inference runtime

User prioritizes SSD/CPU/GPU/NPU/TPU/AIPU adaptability, actual UnbiasedStrata
practices plus other inference techniques, with state and bandwidth central
across workloads. Keep the C core and Modular/Mojo integration target.

1. Native mapping contract/policy: c/runtime_map.h/.c and focused test.
   Rank exact workload/numeric profiles by complete transfer/queue/compute cost,
   staging/memory bounds, compatible state ownership/fences and measured overlap.
   SDK-free Windows/Linux, sanitizers, fast-math and malformed/overflow checks.
   Exit: tested reusable policy with honest separation from runtime actuation.

2. Actual backend profiles and wiring: obtain the existing Mojo/MAX SDK/plugin
   location; implement its native C bridge and stage/link measurement provider.
   Integrate with real workload execution and isolated state leases. Preserve
   current route when no measured profile is qualified. Exit: actual selection,
   reserved resources, completed transfer/state handoff and explicit device proof.

3. Port selected optimizations in distinct measured slices: next-layer prefetch/
   hot residency, chunked prefill, paged/prefix state reuse, continuous batching,
   fused/captured kernels, quantized storage/arithmetic, encoder reuse. Choose
   only techniques compatible with the model's state layout and current backend.

4. Qualification: same target model/checkpoint/numerics and full costs for each
   eligible backend. Text exactness where promised; quality bounds for lossy
   changes; audio/video ordering, timestamps and cancellation; memory/concurrency
   pressure; SSD read/transfer/queue/verification/cleanup latency charged.

Owner of first slice: /root (direct serial). No fake background workers. Do not
mark the broader goal complete from the portable policy or simulated profiles.

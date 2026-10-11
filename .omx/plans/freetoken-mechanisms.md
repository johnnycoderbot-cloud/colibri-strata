# Native FreeToken mechanisms

User authorized adding the researched mechanisms and clarified that Mojo is
the common mapping/dispatch layer, with native MAX, OpenVINO, Vulkan and
device-specific SDKs underneath. Preserve the Colibri C core and state owners.

The repository already has CPU/GPU resident-expert overlap, a shared Vulkan
expert tier with adaptive residency, two-device dispatch and double-buffered
prefill streaming. DSV4 has an opt-in hybrid DMA/CPU split. Do not claim these
existing mechanisms as newly added.

New slice: measured full-cost integer cold-expert split; compiled native Mojo
C-ABI decision/dispatch binding; actual shared Vulkan cold-expert execution
wiring; correct join/order/cleanup; controls and actual paired evidence.
The flat ABI is c/runtime_split.h. Serial upload staging must be charged before
both compute branches; it must not claim new CPU/DMA overlap.
Existing asynchronous prefill staging and residency remain available.

Root owns shared Vulkan wiring, CLI/build integration, engine evidence and
publication. /root/memory_primitives owns runtime_split.c and its native tests.
/root/mojo_dispatch owns actual Mojo ABI implementation, native loader/build
tools and its C/Mojo qualification. /root/architecture owns independent review.
Native in-session fanout; no detached/mock team or imaginary task claims.

Admission: finite achieved costs bound to actual device/phase/shape/numerics,
valid branch capability and memory cap. Every missing expert belongs to one
branch; count is bounded. State remains pinned. Planning publishes once.
Qualified decisions use complete queue/staging/compute/merge costs, not PCIe
marketing numbers. Profile bootstrap and stale/absent measurements must be
explicit. Existing behavior stays available for paired baseline measurement.

Verification: exhaustive small integer minimax oracle and malformed costs,
strict/sanitizer/fast-math C checks; actual installed Mojo compiler and C ABI
replay; shared-tier native Vulkan output/mixed-rank/failure tests; actual engine
replay and same-model paired timings on available hardware. Native Windows
cannot load a WSL shared object: qualify C/Vulkan Windows and Mojo on native
Linux/WSL separately, without combining them into one hardware claim.
Keep final measured defaults conservative if the candidate loses.

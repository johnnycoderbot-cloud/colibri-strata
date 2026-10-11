# Adaptive runtime brownfield map

Existing native engines own family-specific math, KV/recurrent state, routed
expert storage and accelerator kernels. resource_plan.py estimates existing
SSD/RAM/VRAM budgets but must not fabricate bandwidth or physical execution.
Segment/Edge C APIs expose exact state-schema/numeric identities and isolated
sessions; they are opt-in and not linked into ordinary serving paths.

The graph port is a native C process using OpenVINO C APIs. Text/speech compiled
placement is not exposed; upscaling checks actual EXECUTION_DEVICES. AUTO currently
uses an available-device order, not measured costs. Native graph state is opaque
to the current client; migration needs a real adapter state contract.

Recommended boundary: native per-work-unit state/bandwidth mapping policy with
caller-provided exact profiles and resource ownership. Keep all backend execution
and snapshot/restore in adapters. Preserve current family math, greedy MTP
rollback, immutable assets, explicit-device refusal and existing benchmarks.
Mojo/MAX SDK/plugin path remains requested; no runtime integration is inferred
from the user declaring the stack or from generic accelerator API types.

First policy lives in c/runtime_map.h/.c. It prices directed transfers and
budgets, rejects stale/mismatched profiles and unsafe state migration, and
returns a transactional decision. Tests are synthetic and make no speed claim.
Runtime wiring and physical hardware profiling are separate remaining slices.

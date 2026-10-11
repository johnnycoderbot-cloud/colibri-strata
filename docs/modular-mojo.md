# Modular/Mojo integration boundary

The organizing runtime requirement is
[state and bandwidth aware inference](adaptive-runtime.md) across SSD, CPU, GPU,
NPU, TPU and AIPU, with measured per-workload placement.

The user's Modular/Mojo stack is an integration target. The fork now includes a
[native C Mojo/MAX tensor backend](mojo-backend.md), its device probe, compatible
exported graph execution and external safetensor loading. Existing Colibri and
OpenVINO workload paths retain their qualified implementations. Their timings
are not Modular backend measurements.

Keep the Colibri core, model state, speculative verification and storage
scheduler in C. The implemented connection uses the [MAX C API](https://max.modular.com/api/c/).
Deployment-specific kernel/stage routing and state/prefetch integration remain
separate from loading and executing a compatible native graph.

| Component | Intended responsibility |
|---|---|
| SSD/NVMe | Read immutable model assets, stage expert weights and serve the storage tier. |
| CPU/RAM | Host scheduling, target-state ownership, admission and eligible native computation. |
| GPU | Eligible device kernels through the installed Modular backend. |
| NPU/AIPU | Eligible kernels through the user's actual accelerator driver/plugin and compiled model artifacts. |

Storage placement and compute dispatch remain distinct decisions. A runtime
adapter must explicitly connect prefetch, buffers and transfer fences; selecting
a Mojo compute device alone does not implement SSD expert streaming.

Use the installed SDK's current device/plugin identities. [MAX 26.6](https://max.modular.com/releases/v26.6/)
removed its separate NPU class and C constructor; plugin accelerators use the
generic Accelerator route, with backend identity reported separately. Therefore
do not map every accelerator to a guessed GPU/NPU enum or infer AIPU support from
the existence of the generic accelerator API.

The [C type reference](https://max.modular.com/api/c/types/) includes signed
INT4, INT8 and INT16. A dtype identifier does not establish that a particular
kernel/device executes that arithmetic. Each selected backend needs real output,
device, precision, transfer and cancellation checks before claiming acceleration.

## Concrete next handoff

The official pinned native SDK is now installed in the local isolated WSL
qualification environment. CPU graph execution and precision/ownership checks
are recorded in the backend guide. The user's custom accelerator project,
plugins and model artifacts still need to be identified and tested on the actual
bare-metal hosts. A generic runtime device is not an NPU/TPU/AIPU support claim.

Wire qualified stage/link measurements and compatible state leases into the
state/bandwidth mapper, then port actual Strata prefetch/residency and other
inference techniques. Native implementation, an available device backend and a
ported workload remain distinct capabilities. The broader all-family/device
acceleration goal is incomplete.

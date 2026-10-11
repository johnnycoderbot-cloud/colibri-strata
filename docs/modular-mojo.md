# Modular/Mojo integration boundary

The organizing runtime requirement is
[state and bandwidth aware inference](adaptive-runtime.md) across SSD, CPU, GPU,
NPU, TPU and AIPU, with measured per-workload placement.

The user's SSD, CPU, GPU, NPU and AIPU software stack is based on Modular's Mojo.
That is the declared integration target. The published Colibri fork currently
uses its existing C engines and a native OpenVINO C-API graph backend; it does
not yet call a Mojo/MAX backend. Existing measurements belong to those
implementations and do not qualify the user's Modular stack.

Keep the Colibri core, model state, speculative verification and storage
scheduler in C. Connect native compute through the [MAX C API](https://max.modular.com/api/c/)
or a compiled Mojo library with an explicitly agreed C ABI. MAX documents native
model compilation/execution and tensor ownership APIs. Mojo can emit native
[shared libraries or object files](https://docs.modular.com/mojo/tools/compilation/).
Neither route requires replacing Colibri's core with Python model execution.
This is the integration design, not an implemented backend.

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

Locate the user's existing Mojo project or MAX native SDK, pin its version and
identify the installed driver/plugin entry points and compatible model artifacts.
Then implement the C bridge, preserving buffer ownership, bounds, cancellation
and truthful device evidence. Compare the same checkpoint/workload against the
unchanged C baseline, including SSD reads, transfers, verification and cleanup.

The current checkout contains no Mojo/MAX adapter or Mojo source. No compiler or
MAX package was found on the checked Windows PATH, in the hardware venv, or in
the current Kali WSL environment. This does not establish the absence of the
user's stack elsewhere. Its project/SDK location has been requested.

The native OpenVINO port remains published and qualified; the broader Strata
ports and Modular backend integration remain incomplete. Native implementation,
an available device backend and a ported workload are separate capabilities.

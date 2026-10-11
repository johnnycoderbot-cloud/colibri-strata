# Modular/Mojo handoff

User asks to wrap up and identifies their SSD/CPU/GPU/NPU/AIPU software stack as
Modular Mojo. Preserve the native Colibri C core. The published graph port uses
OpenVINO C APIs; a Mojo/MAX connection has not been implemented or benchmarked.

Current handoff: obtain the existing Mojo project/native SDK location and actual
driver/plugin/artifact interfaces. The user has been asked for the path or URL.
Checked checkout, Windows PATH/hardware venv and current Kali WSL do not expose
that runtime. Do not infer absence of the user's stack on another path/host.

After locating it, pin the SDK and choose a real native C boundary (MAX C API or
the user's compiled Mojo library ABI). Implement explicit backend selection,
buffer ownership, dtype/layout admission, asynchronous fences, cancellation and
actual device identity. Integrate storage prefetch with that backend rather than
assuming compute dispatch also supplies SSD streaming. Preserve model state and
target verification in C, and preserve prior measurement settings.

Qualification requires actual full-workload execution on each selected backend
and same-model comparisons including transfer/verification/cleanup costs.
MAX 26.6 uses generic Accelerator/plugin routing rather than the removed NPU
constructor; current SDK headers determine the C ABI. INT4/8/16 type names alone
do not establish per-device kernel arithmetic.

This is a future integration handoff, not a fake active implementation or an
all-hardware completion. Native OpenVINO port qualification remains valid.

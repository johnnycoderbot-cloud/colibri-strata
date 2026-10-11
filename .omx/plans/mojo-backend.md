# Native Mojo/MAX backend

User explicitly asks to add a Mojo backend for bare-metal tests and asks whether
Mojo automatically sees CPU/GPU/NPU/TPU/AIPU. Verify actual runtime-visible
devices; native Mojo does not supply arbitrary drivers or instruction targets.

Chosen path: C model/tensor backend using the real MAX C SDK 26.6.0 (Mojo1.1.0),
plus C CLI and offline model/tensor pack preparation. C owns inputs, bounds,
model lifecycle, output copies and explicit device refusal. MAX executes
compatible compiled graph artifacts through native kernels. Python may prepare
artifacts and invoke the C binary but must not execute production inference.

Files: c/native_mojo_backend.h/.c, c/mojo_runtime_main.c,
c/tools/build_native_mojo.py, c/tools/prepare_mojo_fixture.py, c/coli integration,
focused tests, docs/mojo-backend.md and capability/replay evidence.

Functional gate: native C creates a real SDK context, compiles/initializes a
compatible exported graph, borrows bounded typed tensors, runs inference, checks
output shape/type/source device and copies to owned host outputs. All partial
resources are released and failed calls publish no output. CPU and generic SDK
accelerator are separate explicit selections; missing accelerator is refused.

Probe reports the actual runtime-visible accelerator count and host readiness,
not guessed physical NPU/TPU/AIPU classes. Output placement is narrower
than every-node execution proof. Model compatibility and optional vendor plugins
remain real requirements; existing native checkpoints/OpenVINO IR are not
silently reinterpreted as MAX graphs.

Local compiler/SDK are isolated under /tmp/colibri-mojo-sdk in Kali WSL. Stable
old wheel index returned401; official current install documentation points to
PyPI, whose Modular-owned pinned packages installed successfully. Local CPU
execution is WSL qualification, not a claim of native bare-metal OS testing.
Provide direct native Linux/macOS commands for operator bare-metal tests.

Owner /root, direct serial. No all-family, all-device or speed claim from the
backend smoke fixture. Original state/bandwidth and Strata goals remain active
integration requirements.

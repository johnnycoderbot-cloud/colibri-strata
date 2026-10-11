# Native Mojo/MAX backend

Colibri now has a C model/tensor backend for the native Modular MAX runtime and
its compiled kernels. The new commands execute compatible exported MAX graphs,
including graphs with external safetensor weights, without Python model
inference. Model state must be explicit tensor inputs/outputs; C shadow-copies
inputs, preserves caller ownership and publishes host outputs only on success.

This is an optional backend. Existing Colibri checkpoint formats and OpenVINO
packs are not silently converted into MAX graphs. Tokenizer/decode/workload
adapters and SSD expert paging still need their own integration. The native
Colibri core remains C; Modular supplies the external native runtime/compiler.

## Hardware scope

Mojo cannot automatically use every bare-metal CPU/GPU/NPU/TPU/AIPU. The
[official requirements](https://mojolang.org/docs/requirements/) describe
supported CPU architectures and NVIDIA, AMD and Apple GPU drivers. A device
needs compatible runtime/driver support, a compiler target or working plugin,
and compatible model operators. OS visibility alone is not execution support.

The [MAX C API](https://max.modular.com/api/c/) exposes host and generic
accelerator devices. This backend uses CPU or exact ACCELERATOR.id selectors;
it does not invent NPU/TPU/AIPU identities from a generic accelerator count.
Explicit unavailable accelerators fail and never run on CPU instead.

Current local qualification is CPU execution in Kali WSL on an Intel i9-12950HX;
the Modular runtime reports zero accelerators. This is not a bare-metal OS or
GPU/NPU/TPU/AIPU qualification. The commands below are ready for a supported
native Linux/macOS host. Native Windows Mojo/MAX is
[not currently supported](https://max.modular.com/packages/); use WSL for
development or a supported native OS for bare-metal tests.

## Build and run on a native host

From a source checkout, create an isolated SDK environment:

~~~sh
python3 -m venv .venv-mojo
.venv-mojo/bin/python -m pip install max==26.6.0
.venv-mojo/bin/python c/tools/build_native_mojo.py
python3 c/coli mojo probe
~~~

The qualified compiler is Mojo1.1.0 and native SDK MAX26.6.0. The builder finds
the real headers/libraries through installed package metadata and creates an
owned SDK resource configuration next to the executable. No SDK Python bindings
are imported by the production launcher or the C inference process. The wheel
is used to install/build the native dependencies and to prepare graph artifacts.
Explicit include/library/config paths can qualify other installations.

Run a functional projection/ReLU graph:

~~~sh
.venv-mojo/bin/python c/tools/prepare_mojo_fixture.py --output mojo-test
python3 c/coli mojo run --model mojo-test/model.mef --inputs mojo-test/inputs.json --output mojo-result --device CPU --repeat 20
~~~

On a host with a compatible accelerator, export for that device and select it
explicitly:

~~~sh
.venv-mojo/bin/python c/tools/prepare_mojo_fixture.py --device accelerator --id 0 --output mojo-device-test
python3 c/coli mojo run --model mojo-device-test/model.mef --inputs mojo-device-test/inputs.json --output mojo-device-result --device ACCELERATOR.0 --repeat 20
~~~

Use repeatable --weights file.safetensors arguments for external weight shards;
names must exactly match the compiled graph. The backend keeps each archive
alive until the registry and initialized model are destroyed. Inputs are raw
tensor files with name/dtype/shape declarations in a JSON manifest; relative
paths resolve beside that manifest. Outputs use exclusive files in a new
directory. A failed publication can leave partial files; success requires exit 0.

COLI_MOJO_NATIVE overrides the executable. COLI_MOJO_CONFIG explicitly selects
an existing matching SDK configuration, including custom driver/plugin setup;
otherwise the binary uses its build-generated configuration. SDK resources,
libraries and compiled graph must match. Rebuild on the destination host rather
than copying an absolute-path SDK setup from another machine.

## Bounds, precision and evidence

Inputs/outputs are bounded to 32 tensors, rank 8 and 128 MiB payload/result budgets.
These limits do not cap the SDK's model/compiler/weight allocations. Floating
nonfinite inputs/results are rejected, names and tensor geometry are checked,
and errors preserve output buffers. Calls are serialized by the owner.

Signed INT8 and INT16 inputs are exercised with native integer graphs. Signed
INT4 is a packed low-nibble-first storage codec expanded to INT8 input; the
graph declares INT8. Those fixtures calculate in INT64. This qualifies actual
signed data handling and prevents an assumed raw INT4 layout; it does not claim
native INT4 arithmetic or universal accelerator precision support.

Source output-tensor device placement is checked before copying results to the
host. That proof is narrower than proving every graph node's execution device.
Do not feed it to a scheduler as universal whole-model device qualification.

The CLI reports input reads, compile/load, native runs including input shadows
and output copies, publication, cleanup and total time. External wall time also
includes process/library startup. These synthetic fixture timings are not LLM
tokens/s, an unchanged-Colibri comparison or a speedup claim.

~~~sh
.venv-mojo/bin/python c/tools/build_native_mojo.py --test --sanitize --output mojo-check
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./mojo-check mojo-test/model.mef
~~~

Local exact-reference replays cover F32 projection/ReLU, signed INT4/8/16 data,
external weights and unavailable-accelerator refusal. Actual native SDK checks
cover shape/dtype/byte bounds, nonfinite input, occupied/duplicate/missing
outputs, output budgets and resource ownership under ASan/UBSan/leak detection.
Operator bare-metal GPU/NPU/TPU/AIPU tests remain required before their claims.

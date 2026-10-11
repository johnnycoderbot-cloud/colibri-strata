# Native C graph workloads

The user's Modular/Mojo stack is recorded in the
[native integration boundary](modular-mojo.md). This implemented graph backend
uses OpenVINO C APIs; it has not been connected to Mojo/MAX.

Text generation, speech recognition/English translation and trained 4× image
upscaling now execute inside the persistent C native_graph process. Production
routes call native OpenVINO/GenAI C APIs. WAV decoding, image tiling, BGR tensor
preparation, bicubic auxiliary input/alpha resizing and PNG output are C.

Colibri's existing Python launcher, HTTP server and stream queues remain the
control plane. Model preparation and optional camera/microphone/video container
I/O can use Python packages. Production inference imports no Python model SDK,
NumPy or Pillow. There is no Python inference fallback. The former adapters live
in experimental/strata/reference_stream_models.py for offline comparisons only.

The engine and its clients are C17. OpenVINO's native libraries and its official
C-API wrapper are vendor C++ dependencies behind a C ABI. This port does not
claim that their implementation has been rewritten in C.

## Build and native SDK

Linux: make -C c native-graph. Windows with installed Visual Studio C++ build
tools: python c/tools/build_native_graph.py. Neither engine build needs a
Python ML package or SDK headers. An installed/archive engine is placed beside
the launcher modules; COLI_NATIVE_GRAPH overrides its path.

The qualified dependency is the matching [OpenVINO GenAI 2026.4.1 native Windows
x86-64 SDK](https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.4.1/windows/openvino_genai_windows_2026.4.1.0_x86_64.zip).
Extract it into a user-writable directory. Install the native SDK, rather than
expecting the Python wheel to contain the separate GenAI C library.

Three small C-wrapper extensions are required for text/speech: disable implicit
template wrapping, render the model's own chat template, and set the inherited
Whisper generation limit. They preserve model defaults and exact prompt limits.
The original SDK C DLL lacks these exports. Startup reports that missing
component; it does not run a different inference implementation.

From a source checkout, build the pinned official wrapper in a fresh output
directory. This command downloads only the pinned wrapper sources:

~~~powershell
pwsh -File c/tools/build_native_genai_c.ps1 -FetchSource -SdkRoot C:/sdk/openvino_genai_windows_2026.4.1.0_x86_64 -OutputRoot outputs-native/native-sdk-discovery/native-c-api
$env:COLI_NATIVE_SDK = 'C:/sdk/openvino_genai_windows_2026.4.1.0_x86_64'
$env:COLI_GENAI_C_LIB = (Resolve-Path outputs-native/native-sdk-discovery/native-c-api/openvino_genai_c.dll).Path
python c/tools/build_native_graph.py
~~~

Source revision: ab01d1771e4adaa0e1e08e9d630def170572506c.
The source manifest records all 18 original files' Git blob hashes. The build
preserves the original 153 C exports and adds only the three typed extensions.
The [SDK map](../.omx/research/native-sdk-map.md) records actual build evidence.
Keep this C wrapper, core runtime, GenAI and tokenizers libraries at one matching
SDK version; mixing installations is not qualified.

COLI_NATIVE_SDK scopes runtime/library search paths to the child process.
COLI_OV_C_LIB, COLI_GENAI_C_LIB and COLI_OV_TOKENIZERS_LIB can select explicit
libraries. SDK tokenizers stay in the original SDK runtime directory.
The C executable is build-tested on Linux and Windows; real SDK/model replay
below is Windows. The wrapper rebuild script uses MSVC. Linux/macOS native
dependency builds need the same published patch applied to the matching official
wrapper and remain unqualified here. This is a port/qualification gap, not a
workload that C cannot execute.

## Run

Prepared immutable packs and existing CLI/API routes are unchanged:

~~~sh
python c/coli run --model models-stream/text --device GPU --ngen 24 "Explain gravity."
python c/coli transcribe --model models-stream/speech-stateful --device GPU recording.wav
python c/coli transcribe --model models-stream/speech-stateful --device CPU --translate recording.wav
python c/coli upscale --model models-stream/camera --device GPU photo.png --output enhanced.png
python c/coli upscale --model models-stream/camera --device GPU input.mp4 --video --output enhanced.mp4
~~~

See [workload and stream contracts](runtime-expansion.md) for preparation,
authentication, capture commands, input bounds and timestamp/audio handling.
Inference is trained super-resolution; bicubic supplies an auxiliary model input
and alpha, rather than replacing the network. Frames use the same native engine.

Explicit CPU/GPU/NPU selection must be available and compile successfully. AUTO
tries available GPU, NPU then CPU and records failures; it does not benchmark
device speed. Text/speech report constructor selection with
execution_devices_verified:false. Super-resolution checks EXECUTION_DEVICES
before and after each inference. NPU requires a compatible model/device/plugin;
TPU/AIPU full-model adapters still need ports and physical qualification.

Cancellation or a deadline kills and drains the owned C process, discards its
result and lazily reloads for the next request. Speech's current native GenAI API
has no interrupt callback; the process boundary provides bounded cancellation.
Admission remains an estimate, not a reservation or an OS memory cap.

## Verification

~~~sh
make -C c native-graph native-graph-check
python -m unittest discover -s c/tests -p test_stream_runtime.py
python -m unittest discover -s c/tests -p test_runtime_gateway.py
~~~

SDK-free gates cover ABI ownership/failure paths, model hashes/path aliases,
media tensor parity, framed transport, deadline/cancellation/cleanup and native
PNG bounds, including Adam7, invalid palette indexes and excess decoded padding.
Offline Python oracle tests use the separate graph-reference extra.

Real text, Whisper tiny INT8 and trained SISR1032 replays exercise production
routes with Python ML imports blocked. They establish functioning native paths,
not a new throughput or power claim. Existing sequential MTP speed measurements
are unchanged by this graph port.

[Recorded native qualification](../.omx/research/native-graph-qualification.json):
six CPU/GPU HTTP replays, exact text/token and three image-fixture comparisons,
English translation, and a three-frame trained GPU video replay with exact audio
payload and preserved timestamps. [Independent integration review](../.omx/research/native-integration-review.md).

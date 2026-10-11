# Native C graph inference port

User requires native execution like Colibri. The native text engines and Strata
MTP remain C. Existing text_graph, speech/English translation and trained 4x
image/video/camera inference must move from Python model adapters to a persistent
C executable using native OpenVINO/GenAI C APIs. These are not inherently
unsupported workloads; SDK/port/device availability are separate states.

Python may retain Colibri's existing control-plane work: launcher, HTTP framing,
model preparation, admission, stream queues and optional capture/video container
I/O. It must not import model SDK bindings or NumPy/Pillow to execute/preprocess
the model in production inference. No Python inference fallback. Preserve current
manifest bounds, immutable asset hashes, explicit-device refusal, cancellation,
resource close and learned-model image output. Vendor native SDK implementations
can internally use C++; our inference clients, media tensor prep and engine are C.

Native in-session ownership (real live agents; no detached/mock team):

- /root/architecture: official SDK/version/compiler discovery and research map.
- /root/parallel_heads: C backend ABI, native SDK client and fake-SDK checks.
- /root/memory_primitives: pure C WAV/image tensor helpers and focused C tests.
- /root: native process/manifest/PNG framing, control-plane client wiring, build,
  real model replay, integration review, documentation and publication.

First settle typed C ABI and SDK. Implement a bounded persistent process with
raw request bytes and JSON/raw results; perform model/tokenizer/media preparation
in native code. Wire GraphEngine to the native client, retain queue/gateway tests,
and add refusal/timeout/cleanup tests. Build Linux model-free and Windows native
SDK paths. Replay all three prepared real packs on available CPU/GPU and verify
their actual results/device evidence. Do not claim NPU/TPU/AIPU physical execution
without hardware evidence. Publish supported native ports and label missing
device/SDK/port components precisely; preserve original model assets.

Completion requires production routes invoking the C executable, no Python model
SDK imports in those routes, actual neural upscaling, English speech/text replay,
bounded protocol/teardown behavior, reviewed source and published evidence.

# Colibri graph workloads

The actual family registry, `coli` CLI and authenticated gateway now load compact
text, speech and trained 4× upscaling packs. These production adapters are separate
from experimental future-token heads. They now execute in a persistent C engine
through native SDK C APIs. See [native build and SDK setup](native-graph.md).

| Family | Workload | Implemented backend |
|---|---|---|
| `text_graph` | Deterministic completion; stateless full-history chat | OpenVINO GenAI CPU/GPU/compatible NPU |
| `speech` | Short-window recognition; speech-to-English translation | OpenVINO GenAI CPU/GPU/compatible NPU |
| `camera_upscale` | Trained 4× image, video and explicit camera upscaling | OpenVINO CPU/GPU/compatible NPU |

Text inference can run on NPU, TPU and AIPU when the model, operators, precision
and compiler/runtime support it. **This production graph adapter uses OpenVINO.**
Explicit devices must be available and compile successfully; an unavailable NPU
fails instead of running on CPU. `AUTO` tries available supported devices and
reports selection/compilation failures; it does not measure the fastest device.

Full-model TPU/AIPU text, speech and upscaling adapters are not implemented or
qualified here. Existing JAX TPU **head** execution and the AIPU compiled-image
probe are separate diagnostics. Compatible NPU hardware also still needs a
physical model replay. Do not infer full-model hardware support from those probes.

Native Windows GenAI **2026.4.1** speech replay now generates correct text and
shuts down cleanly on CPU and GPU. The old Python interpreter teardown failure
does not gate this C path. Other SDK/platform combinations require qualification.

## Setup and commands

Use Python 3.12 in a virtual environment from a source checkout:
Build the native executable and configure the matching native SDK first, following
the [native setup guide](native-graph.md#build-and-native-sdk).

```sh
python -m pip install -r requirements-runtime.txt
python c/coli prepare-runtime speech --output models-stream/speech
python c/coli prepare-runtime camera_upscale --output models-stream/camera
python c/coli prepare-runtime text_graph --output models-stream/text
python c/coli run --model models-stream/text --device GPU --ngen 64 "Explain gravity."
python c/coli chat --model models-stream/text --device GPU
python c/coli transcribe --model models-stream/speech recording.wav
python c/coli transcribe --model models-stream/speech --device NPU --translate source.wav
python c/coli upscale --model models-stream/camera --device GPU photo.jpg --output enhanced.png
python c/coli upscale --model models-stream/camera --device GPU input.mp4 --video --output enhanced.mp4
python c/coli serve --model models-stream/text --device GPU --api-key YOUR_KEY
```

Preparation explicitly downloads pinned public artifacts and records SHA256
identities, refusing existing output directories. Startup never downloads models.
Packs must remain immutable while in use. Speech uses stateful Whisper tiny INT8;
text uses Qwen2-0.5B-Instruct INT4; upscaling uses pretrained Open Model Zoo SISR1032
FP16. Artifact precision does not establish each accelerator kernel's arithmetic.
FP16 is floating point, not signed INT16. The child C process loads native SDK
libraries; the Python production adapter imports no model bindings.

Camera packs default to an `f32` inference hint. An explicit manifest option
`options.inference_precision: "f16"` enables a floating-point precision hint on
CPU/GPU plugins; it does not guarantee every kernel's precision. The local trained
GPU f16 replay differed from f32 by mean 0.131 and maximum 6 pixel values, below
the fixture's declared bounds of 0.5 and 8. This is a fixture check, not universal
image-quality qualification or signed INT16 execution.

`--memory-budget-mib` defaults to 1024; `--ram N` is an alternative total estimate
budget in GiB. `plan` checks hashes/admission estimates; `doctor` also compiles the
selected backend. Queues and result staging count against admission. Estimates do
not reserve or cap physical memory; backend allocations can fail. Native-only
VRAM, expert and tuning flags are refused when specified on graph commands.

## Media and capture

CLI images accept PNG/JPEG/WebP; the native API accepts 8-bit PNG, including
palette/alpha and Adam7 inputs. The trained model receives a
low-resolution BGR tile and bicubic 4× input; results use the **network output**.
Overlapping padded tiles support arbitrary shapes up to 349,525 source pixels
(5,592,405 output pixels), including 640×480 camera frames. Larger sources fail;
1080p input is beyond this compact default. Alpha is retained separately.

Video export processes every frame, retains source presentation timestamps to a
1/90,000-second tick, and remuxes audio streams. MP4 uses H.264; other suffixes use
Matroska. Inputs need one video stream, valid frame timestamps and compatible audio
codecs. Frames are enhanced independently; temporal/motion-aware restoration is
not implemented. Outputs use exclusive creation. Interrupted export leaves a
partial file and fails explicitly.

Capture only occurs after explicit commands:

```sh
python c/coli transcribe --model models-stream/speech --microphone --seconds 5
python c/coli upscale --model models-stream/camera --device GPU --camera 0 --frames 60 --output camera-frames
```

The microphone command records one window. API audio sessions accept multiple
windows. Accepted audio results are retained until polled or cancelled; full queues
return a retryable error. Camera sessions replace oldest pending frames and report
their exact identities. Full event queues require clients to drain before retrying.
No microphone or camera was opened during development checks.

Polling consumes events before the HTTP response is sent: delivery is at most once,
and a lost poll response cannot be replayed. Push retries after a lost acceptance
receipt are not idempotent; already-accepted sequence numbers are rejected. Sessions
are in-memory and do not survive process restart.

## HTTP contracts

Routes reuse Host checks, authentication, request deadlines, body framing and
connection caps. Each served pack exposes its own workload. Backend operations
are serialized; shutdown drains work before destroying the model.

| Route | Request / behavior |
|---|---|
| `POST /v1/chat/completions` | `model`, string-content `messages`, `max_tokens`, `stream:false` |
| `POST /v1/completions` | `model`, string `prompt`, `max_tokens`, `stream:false` |
| `POST /v1/audio/transcriptions` | JSON `model`, `audio_base64`, optional `language`; or multipart `file`, `model`, `language` |
| `POST /v1/audio/translations` | Same audio input; automatic source language, English target |
| `POST /v1/camera/upscale` | `model`, `png_base64`; returns enhanced PNG, dimensions and device evidence |
| `POST /v1/runtime/streams` | `model`, `kind:audio` or `kind:camera`, `queue_capacity`, optional audio `options` |
| `POST /v1/runtime/streams/{id}/push` | `model`, increasing `sequence`, nondecreasing `timestamp_ms`, `payload_base64`, optional `final` |
| `GET /v1/runtime/streams/{id}/events` | Bounded `max_events`/`wait_ms`; results, drops, errors and final events |
| `POST /v1/runtime/streams/{id}/cancel` | `model`; discard queued and stale in-flight results |
| `POST /v1/runtime/streams/{id}/close` | `model`; `closed:false` means drain needs a retry |

Text requests supply full history; backend client state is never shared. Defaults
cap prompts at 2,048 tokens and outputs at 256. Sampling controls, tools and token
SSE are rejected. Completion-token usage and EOS/token-limit finish reasons are
not inferred from text.

Speech requires finite 16kHz mono PCM16/float32 WAV, strictly shorter than 30 seconds.
Multipart supports `response_format=json`. Source language can be English or
automatic; translation targets English. Each payload is independently decoded.
There is no VAD, cross-window decoder state, partial hypothesis revision or word
alignment. `segments` is empty; session timestamps identify input windows.

## Storage and evidence

Native Colibri engines retain SSD/NVMe, RAM and GPU expert placement. Graph packs
are read, verified, compiled and held by the backend; this compact lane does not
page arbitrary graph layers from SSD into every accelerator. Device selection
changes eligible compute, not storage policy.

The [previous Python reference replay](../.omx/research/stream-model-replay.json)
records pinned assets and earlier warm timings. The [native port](native-graph.md)
has separate real-model checks; those older timings do not measure this C port.
Camera execution checks
`EXECUTION_DEVICES` before and after inference; CPU/GPU outputs agree within one
pixel on the fixture. GenAI exposes no compiled-model device property here:
text/speech report explicit constructor selection with
`execution_devices_verified:false` and `executed_device:null`.

GPU execution does not guarantee lower elapsed time. Small-image end-to-end
medians here were about 151ms CPU and 169ms GPU, including PNG handling and
transfers. These measurements do not establish real-time camera speed, lower
power or universal speedup. Replay tests do not replace a quality benchmark.

Tests cover malformed manifests/media, aliases/FIFOs, admission, queue saturation,
retry/order/drop identities, cancellation/drain/client disconnect, duplicate MIME
parameters, HTTP framing/auth, video timestamps/audio and native packaging.
Windows/Linux CI uses a real CPU IR fixture; trained replays are physical checks.

```sh
python c/tools/check_runtime_gateway.py --model models-stream/text --device GPU --report text-replay.json
python c/tools/check_runtime_gateway.py --model models-stream/camera --device GPU --input frame.png --report camera-replay.json
```

Remaining hardware/model qualification: [issue #6](https://github.com/johnnycoderbot-cloud/colibri-strata/issues/6).

# Streaming runtime implementation

User said to proceed with issue6 and selected **English speech output and camera,
image and video upscaling**. They also requested text inference and actual hardware
acceleration. Public accelerator labels remain CPU/GPU/NPU/TPU/AIPU.

Deliverable: registered graph model families used by the real coli CLI and existing
authenticated HTTP gateway, compact trained speech and super-resolution adapters,
bounded persistent stream sessions, explicit device/fallback reporting, replay and
opt-in capture commands. Existing native model tiers and defaults are preserved.

Model choices: multilingual Whisper tiny INT8 OpenVINO IR for recognition and
speech-to-English translation; trained single-image-super-resolution-1032 OpenVINO
IR for 4x upscaling; Qwen2-0.5B-Instruct INT4 for compact accelerated text.
Artifacts are explicitly prepared with pinned revisions/hashes,
not downloaded on server startup or bundled into source control.

Native slices:

1. /root/memory_primitives owns c/stream_runtime.py and its tests: verified manifest,
   memory estimates/admission, model ownership, bounded audio/camera sessions,
   monotonic ordering, cancellation/drain and truthful events.
2. /root/parallel_heads owns c/stream_models.py,c/stream_media.py and adapter tests:
   real lazy Whisper/OpenVINO and trained super-resolution, strict WAV/PNG handling,
   explicit supported device execution, output validation and cancellation.
3. /root owns registry/resolve_model, CLI, existing API routes/stream polling, model
   preparation, optional capture, installation closure, docs, integration tests and
   artifact-backed physical CPU/GPU/replay tests.
   c/tests/test_runtime_gateway.py was delegated to /root/memory_primitives after
   its runtime slice completed; the native agent remained the real owner.
4. /root/architecture reviews boundaries and final implementation independently.

Verification: manifest tampering/paths/caps; queue full/order/cancel/late output;
media truncation/rates/channels/nonfinite/pixel limits; actual trained recognition
and upscaling replay; model registry->CLI->existing authenticated API integration;
wrong endpoint/backend refusal; native family/package/build regressions; targeted
CI and installed module closure. Mock/fake adapters qualify control flow only.

No microphone/camera is opened during development checks. Capture is only through
an explicit live command. Real NPU/TPU/AIPU qualification stays per compiled model;
absence is reported instead of hidden CPU substitution. Do not claim low-power,
real-time FPS, arbitrary-language translation or every precision from a replay test.

Implemented: three graph families, actual CLI and authenticated gateway, pinned
preparation, bounded queue ownership, media validation, text state isolation,
video/audio timestamp transport, optional explicit capture, package closure, CI.
Independent implementation review approved after registry budget/hash lookup,
HTTP error-class identity, MIME aliases, ignored native flags and palette-alpha
findings were corrected. Physical replay/SDK process-exit qualification remains
the final publication gate; TPU/AIPU full-model adapters remain explicit follow-up.

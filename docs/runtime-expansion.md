# Extending the existing runtime to streaming workloads

Colibri already manages weights through GPU VRAM, CPU-addressable RAM and storage.
Routing heat, cache budgets, prefetch, placement policy and measured costs influence
where experts live and execute. This fork preserves that existing memory hierarchy.
There is no architectural rule limiting the runtime to text token generation: the
upstream family registry already includes text, vision/image and decision engines.

Speech recognition, translation and camera effects can become additional model
families. They need modality adapters and streaming scheduling as well as a compatible
compute backend. The current fork has tested standalone head runners; these streaming
families have not yet been implemented.
Tracked implementation: [streaming-runtime issue #6](https://github.com/johnnycoderbot-cloud/colibri-strata/issues/6).

## Proposed model-family contracts

| Workload | Input/state | Output | Required validation |
|---|---|---|---|
| Speech recognition | PCM chunks, sample rate, frontend features, VAD and decoder state | Partial/final transcript and time spans | Offline/live parity, word error, chunk boundaries, silence, cancellation |
| Live translation | Source language, target language, stable transcript spans and encoder/decoder state | Translation with revision/commit boundaries | Language/sequence identity, held-out quality, incremental context and latency |
| Camera effects | Frames, color space, dimensions, capture timestamps and temporal state | Processed frames/masks | Layout/color correctness, bounded queues, frame drops, output quality and latency |

Capture is a separate opt-in layer. Microphone/camera access should occur only when
the user starts the relevant live pipeline; tests can replay fixed licensed fixtures.

## Resource and scheduling design

1. Register each model's artifact identity, tokenizer/frontend, shapes, state bytes,
   supported precision, modality and backend constraints.
2. Reuse memory budgets, ownership, staging and cache lifetimes. Compact dense graphs
   can fit an accelerator entirely; large MoE text models continue using expert tiers.
3. Add persistent streaming sessions, bounded queues, cancellation/drain and timestamp
   contracts. Real-time camera queues can discard stale frames; text/token state must
   preserve causality. Speech/translation segments need explicit revision boundaries.
4. Select supported CPU/GPU/NPU/TPU/AIPU lanes according to model compatibility,
   available memory, transfer costs, deadline and measured power/latency. OpenVINO
   provides a native supported-NPU route; JAX supplies an explicit TPU route; AIPU
   execution uses its installed SDK and qualified compiled artifacts.
5. Keep requested-device validation strict. A diagnostic NPU/TPU/AIPU check cannot
   succeed by silently running on CPU. An automatic production policy may choose a
   documented fallback, but must report where the work actually ran.
6. Admit each model/backend only after reference outputs, state/failure tests,
   live/replay equivalence and physical placement/power measurements.

The next integration is a generic model-graph executor connected to the existing
family registry and engine/backend ownership. The present head-specific exporters
do not yet load arbitrary speech or vision graphs. Preprocessing, streaming state,
decoder behavior and application endpoints must land alongside their model adapters.

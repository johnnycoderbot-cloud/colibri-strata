Extend Colibri's existing model-family registry and tiered runtime to speech
recognition, incremental translation and camera effects. Keep generic CPU/GPU/
NPU/TPU/AIPU labels; reuse weight ownership, budgets and storage tiers.

Acceptance:

- Register model artifact/frontend/tokenizer/state/backend capability contracts
  and execute arbitrary compatible model graphs, rather than only future heads.
- Provide bounded audio/frame queues, persistent streaming state, timestamps,
  cancellation/drain, transcript revision/commit and stale-frame policies.
- Add speech recognition and translation family adapters plus a camera-effects
  adapter with named trained artifacts and reproducible fixtures.
- Select only supported precision/shape/backend combinations. OpenVINO is an
  explicit NPU route, JAX a TPU route, installed SDKs supply qualified AIPU work.
- Validate reference outputs, ASR/translation/effect quality, stream boundaries,
  latency, power and actual execution devices. An absent-device test cannot
  silently pass through CPU substitution.
- Capture devices are opt-in; fixture replay tests need no microphone/camera access.

See docs/runtime-expansion.md. Current standalone head runners and vendor-artifact
checks are useful groundwork; streaming modality handlers remain unimplemented.

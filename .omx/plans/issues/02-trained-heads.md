The offline trainer/exporter exists, but there are no bundled trained real-model
head weights or validated live hidden-state taps. Add per-engine trace capture
and reproducible training/evaluation for the same checkpoint used by inference.

Acceptance:

- Bind each trace/pack to pinned checkpoint, tokenizer digest, quantization,
  hidden-state tap and geometry; train offset2+ labels without crossing sample,
  padding or EOS boundaries.
- Produce real model-specific packs; keep all base weights/router semantics fixed.
- Save train/held-out corpus identities and raw conditional acceptance by horizon.
- Validate compiled CPU/GPU bank predictions against training/export references.
- Report bank time, verify time, total request latency and comparison with native
  MTP/lookup; low training loss or synthetic perfect guesses are insufficient.
- Compare F32/int8 head quality before adopting compressed trained heads.

See experimental/strata/README.md and docs/strata-benchmarks.md. No checkpoint
downloads or costly training are part of the existing bootstrap deliverable.

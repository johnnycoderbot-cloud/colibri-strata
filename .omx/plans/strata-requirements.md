# Colibri-Strata fork requirements

User request (2026-10-10 America/New_York): create a GitHub fork of Colibri that
pursues Strata-inspired inference optimizations, including independent future-token
guessing heads. User explicitly selected interfaces across **all model families**.

Source baseline: JustVugg/colibri `bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850`.
Reference: Niko1221/Strata `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`.
Destination: authenticated personal account johnnycoderbot-cloud, colibri-strata.

Initial deliverable: an actual GitHub fork preserving upstream history and Apache
license, a tested portable experimental core for independent parallel heads,
greedy transactional verification, INT8 block storage, hot-expert selection, and
two-slot prefetch fences; an offline head trainer/exporter; CI, integration matrix,
and concrete tracked implementation work for real engines and hardware.

Non-goals for this bootstrap: modifying model router semantics, making unmeasured
throughput claims, pretending synthetic tests certify production engines, inventing
trained model-specific weights, downloading huge checkpoints or spending money on
training. Existing engine defaults stay as upstream until a family adapter is
validated. INT8 KV is lossy and requires model quality evidence before enabling.

Decision boundaries: fork creation and commits are authorized; use the personal
authenticated account and descriptive repository name; preserve upstream license;
publish only repository project files. Separate baseline capabilities, experimental
portable primitives, and remaining engine/backend integrations in every status.

Acceptance: upstream-derived GitHub repo exists with published code; all added C
behavior tests pass serial/OpenMP and sanitizer runs; trainer smoke tests pass;
CI covers added portable code; family coverage is explicit; real-model performance
remains unverified with next integration tasks documented.

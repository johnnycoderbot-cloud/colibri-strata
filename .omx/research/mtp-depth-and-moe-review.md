# Draft depth and MoE prefetch handoff

Direct source/evidence review by `/root`: **approved** for this CPU measurement
slice. No new independent review was performed; previous pretrained-cell math
and target-state reviews remain applicable because this slice changes telemetry
and request completion, not model math.

## Outcome

- Qwen3.5-0.8B maximum two versus one: paired median 1.0125547; maximum three
  versus one: 1.0009079. Each study has fifteen alternating pairs, the same
  executable/sidecar/precision/gate, and three exact numeric checks. Graph-search
  workloads regress in both studies. Keep maximum one as the general default.
- Existing OLMoE CPU PILOT off/on with eight cached experts per layer: 0.9861831;
  with sixteen: 1.0062937. Each study has fifteen pairs and three exact numeric
  checks. Keep PILOT off by default. Increased hit rates alone do not prove speed.
- All sixty new timed outputs match their numeric oracle; all twelve separate
  token-ID/logprob comparisons match. All measured engine/model/prompt identities
  remain unchanged before/after. Both active MTP arms require real cell, actual
  depth and source-specific acceptance evidence.

## Review checks

OLMoE emits PROF after DONE. The new single-slot collector consumes both within
the originating request and labels its rate as whole-request completion speed.
Queue drain includes unused expert reads before timing ends. Real worker loads
and completed queue items are recorded separately; missing/unfinished evidence
fails qualification. Its held-read concurrency test passed. Phase read durations
can overlap computation and are not additive request-time components.

Collectors retain failure/partial reports, close owned processes, guard numeric
frames and bounds, refuse existing output paths, and use explicit argument arrays.
The new collector exclusively claims its output descriptor before measurement.
Both prefetch arms have the same fixed cache budget and differ only in PILOT.
Both depth arms differ only in their maximum draft count.

The staged native-filesystem model matched the source manifest
`485567877718d9a145129bb14412d77017ac35d36e248318d4dc23430c5eaf3f`.
After successful measurement its temporary copy was removed after exact path
and inventory checks. Original assets and all raw/curated reports remain.

## Verification

- 47 Python benchmark-driver tests passed on the host.
- Complete `make -C c strata-native-check` passed under Linux, plus the legacy
  dense-batch and OLMoE serve-framing tests.
- ASan/UBSan passed for OLMoE cache/drain concurrency and the legacy dense-batch
  test that failed published CI.
- Published CI failures were isolated to an older dispatch-counter test assuming
  grouped projections were off and POSIX-only feature declarations hiding macOS
  APIs. The legacy test now explicitly isolates its old dispatch contract, and
  the two new kernel tests request Darwin declarations on Apple platforms.
- Public reports retain every pair, exact numeric evidence, hashes and actual
  draft/prefetch telemetry. Private model paths and unrelated hardware probes
  are excluded from publication.

## Limits and next work

This slice qualifies warm native-filesystem CPU requests. It does not qualify
cold SSD latency, new GPU DMA, all-family heads or full INT4/INT16/NPU/TPU/AIPU
execution. Fresh remote Windows/macOS CI results remain pending. Larger-model
GPU overlap/residency and additional family adapters remain on the broad roadmap.

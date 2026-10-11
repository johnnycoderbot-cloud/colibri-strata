# Draft depth and larger MoE qualification

User approved: compare sequential draft maxima2/3 with currentmaximum1, retain
the fastest verified default using the existing adaptive gate, then test a
larger MoE model where expert prefetch/cache costs are meaningful.

Root owns this serial measurement slice; no detached or parallel CPU actors.
Qwen3.5-0.8B native8/IDOT, same compiled engine/sidecar/prompts/threads8,
64-token cap, >=5 alternating pairs per each of3requests. Botharms enableMTP;
only maximumdepth differs. Gate remainsenabled. Capture actualchosen depths,
source-specific acceptance, drafttime/targetforwards, wall/TPS and exactnumeric
output parity; retain allasset/binary hashes and failed/partial reports.

First add minimal telemetry and same-binary depth comparison to the existing
collector with meaningful regression coverage, then freeze and time max2/3.
Choose newdefault only if gain is clear acrossrequests; otherwise retain1.
Then measure prepared OLMoE1B7B nativecontainer onCPU with fixed expertcache
budget and PILOT0 versus1. Distinguish existingColibri prefetch from newports;
do not attribute it to an unimplemented GPU/NPU backend or assume warmOS cache
is coldSSD. Native targetIDs/logprobs must match before performance claims.

Publish reviewed code/evidence/docs and keep broader device/family goal pending.

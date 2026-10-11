# Pretrained sequential MTP inside Colibri

User clarification: sequential Strata-style drafting is acceptable. The earlier
requirement that future-token heads be independent and parallel is superseded.
The objective remains measured Colibri inference acceleration, not a separate
Strata engine. Existing independent heads remain an opt-in research path.

Current native checkpoint Qwen3.5-0.8B contains an authentic trained MTP cell;
the Colibri converter currently discards it. Use that cell instead of fitting
new low-rank future heads. Existing qwen38 already has model-specific sequential
MTP; reuse the proven state/verification ideas after validating family geometry.

1. /root/architecture: exact upstream semantics, target/MTP frontier and prefix
   contracts; then independent review. No source writes during discovery.
2. /root/memory_primitives: explicit opt-in MTP sidecar conversion and tiny
   fixtures; never overwrite original target assets or silently drop tensors.
3. /root/parallel_heads: pretrained Torch reference/quality, then separate native
   MTP CPU cell and numerical oracle tests. No retraining or linear substitute.
4. /root: qwen36 state capture, sequential proposal/verify/settle ownership,
   prefill/prefix/cancel integration, CLI and benchmark, docs and publication.

Acceptance: exact cell reference parity on tiny fixtures; causal KV only;
accepted target output/logprob equality, first/mid rejection, full acceptance,
budget, EOS, cancellation and subsequent continuation, pin/prefix reset.
Native one-cell and deeper sequential windows benchmark against same target
binary MTP-off with existing lookup preserved; three fresh prompts and at least
five alternating pairs, all costs included. Hardware and family qualification
remain explicit. Success means a real measured benefit; correctness alone is
not acceleration. Stop other compute before model performance runs.

Current verification completed:16converter,10C+74Pythoncombinednative gates,
independent strict/scalar/ASan+UBSan cell/oracle and8/32frontier tests. Real model
actualdepths1/2/3 matchexacttargetIDs/LP/hidden; forcedrealcellrollbackcases,
pin/prefix/sampling/EOS/cancellation+continuation andprecision refusals pass.
Generateddependency8 andVKlink2 checks pass. Independentreviewapproved.
SamebinaryMTPoff/on:15pairs,3testrequests,64token cap,8threads;pairedmedian
1.1313256 (+13.1%),41.241→46.657 medianarmTPS,all15wins;fullwall1.1221;
allnumericcomparisonsmatch,allweights/enginehashesstable. Wholefork-vs-stock
comparison also completed; publication is next.

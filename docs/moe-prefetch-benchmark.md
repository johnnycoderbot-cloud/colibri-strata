# Native OLMoE expert-prefetch qualification

This compares **existing Colibri `PILOT=0` versus `PILOT=1`** in the same native
OLMoE CPU executable. PILOT predicts the next layer's experts while the current
layer computes, and its background worker loads them into the RAM expert cache.
Actual routing still determines the experts used by the model. This is a
qualification of Colibri's existing lookahead, separate from the new qwen36
pretrained MTP and CPU projection ports.

## Measurements and default decision

| Experts cached per layer | Paired request-speed change with PILOT | Bridge | Python | Graph search | Mean cache hit, off → on | Peak request RSS |
|---|---:|---:|---:|---:|---:|---:|
| Eight | -1.4% | -2.3% | -0.7% | -1.5% | 45.2% → 48.9% | 2.58 GiB |
| Sixteen | +0.6% | -0.6% | +0.7% | +1.1% | 62.9% → 67.5% | 3.33 GiB |

Each row is its own fifteen-pair off/on trial with an identical fixed budget in
both arms. These are not paired comparisons of eight versus sixteen cached
experts. All six separate token-ID/logprob/output comparisons matched exactly;
all thirty timed outputs matched their numeric oracle. Model, prompt and binary
hashes stayed unchanged before/after both trials.

The worker performed 11,703 real expert loads in the eight-slot timed candidate
requests and 10,295 in the sixteen-slot requests. Every queued item completed
before its originating request ended. Higher cache hits did not yield a clear,
consistent speed gain, so **PILOT remains off by default**. The original
Qwen3.5-0.8B pretrained-MTP gain remains independently measured.
[Eight-slot evidence](../experimental/strata/results/olmoe-prefetch-cache8.json),
[sixteen-slot evidence](../experimental/strata/results/olmoe-prefetch-cache16.json).

## Reproduction and timing scope

The prepared checkpoint is AllenAI OLMoE-1B-7B-0125-Instruct: sixteen layers,
64 experts per layer, eight selected per token. Its source GGUF is pinned at
`allenai/OLMoE-1B-7B-0125-Instruct-GGUF@2ac1d27317927518c9ef7bd99f91f3f1e4ee288d`,
SHA256 `4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f`.
The existing conversion has native signed INT8 expert weights. Both arms use
eight CPU threads, the same per-layer cache capacity, `IDOT=0`, `FUSED3=0`,
`HOT=0`, `WIDE=1`, `SMOOTH=0.3`, `CONF_LIMIT=0.92`, and no GPU or prefix reuse.
Three distinct instruction requests have a 64-token output cap and five
alternating timed pairs each, following separate numeric checks and warmups.

The model was staged on WSL's native filesystem after reads through the Windows
shared-folder bridge dominated a smoke request. Source/staged manifests were
identical. Preflight hashing warms the OS page cache; staging/startup are
outside timed requests. This measures a warm file-backed RAM cache, not cold
physical SSD latency or GPU PCIe overlap.

OLMoE's `DONE` rate includes prompt processing and generation. The collector
calls it **whole-request completion tokens/s**, and consumes the subsequent
untagged `PROF` within the same single-slot request. Disk phase durations include
overlapping worker reads; they cannot be added to compute durations to reconstruct
request time. Isolated prefill/steady-decode rates are unavailable.

Single-slot request completion now drains its prefetch queue before timing ends.
This charges unused work to its originating request and prevents it spilling into
the other benchmark arm. Per-request telemetry records enqueued items, completed
items and real worker loads; the collector refuses unfinished queues, absent
execution evidence, output changes, or changed engine/model/prompt assets.
The queue drain has a held-read concurrency regression test.

```sh
make -C c olmoe
python c/tools/native_moe_prefetch_bench.py --engine c/olmoe \
  --model NATIVE_OLMOE --prompts three-olmoe-prompts.json \
  --output NEW_RESULT.json --threads 8 --cache-per-layer 8 --repeats 5
```

Run on the same filesystem in both arms. Prompt text and tokenizer hashes are
preserved with the result. The prompts use the prepared checkpoint's chat
template, including its configured BOS marker; model assets are unchanged.

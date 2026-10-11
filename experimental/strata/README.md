# Offline independent future heads

`heads.py` trains low-rank linear heads from **frozen target hidden-state traces**.
There is no model download and no bundled trained checkpoint. Training uses PyTorch;
the C primitives and trace/pack validators are dependency-free.

An engine adapter must first capture a named immutable hidden vector at position t
and record actual future tokens from the **same tokenized sequence**. Do not cross
documents, samples, EOS or padding. Missing future positions are `null`, never a
token from the next training sample. Real-engine trace capture is on the roadmap;
the format does not substitute for a validated hidden-state tap.

First line of the UTF-8 JSONL trace:

```json
{"schema":"colibri.strata.hidden-trace/v1","family":"qwen38","checkpoint_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tokenizer_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","hidden_tap":"synthetic_fixture_readout","quantization":"fixture-f32","hidden_size":2,"vocab_size":4}
```

The hashes above are **synthetic examples**, not checkpoint identities. Replace
them with the actual model revision and SHA256 of its tokenizer before real training.
Each later line:

```json
{"hidden":[1.0,0.0],"future_tokens":[0,1,2,3]}
{"hidden":[0.0,1.0],"future_tokens":[0,3,1,null]}
```

`future_tokens[0]` means offset1 / x_(t+1). The target already supplies it, so
default learned offsets 2,3,4 use indices 1,2,3. All heads read the same hidden
vector; training runs independent down/up matrices together, without a prior
head's guess as input. Choose ranks and batch sizes that fit memory.

```sh
python -m pip install -r experimental/strata/requirements-training.txt
python experimental/strata/heads.py --trace target-trace.jsonl \
  --out target-heads.json --export-c target-heads.h \
  --offsets 2,3,4 --rank 16 --steps 1000 --batch-size 8
python -m unittest discover -s experimental/strata -p 'test_*.py' -v
python experimental/strata/export_smoke.py  # requires torch and gcc/clang with OpenMP
```

Outputs must be new files. The JSON pack binds family, checkpoint, tokenizer,
quantization, hidden tap and geometry. An adapter must compare **every identity
field**, not just shape, before use. `validate_pack(pack, expected_identity)` is the
Python guard. C export includes matching metadata strings and a `strata_trained_bank`
for `c/strata_heads.h`; use `-Ic` at compile time. This is a trusted build-time export,
not a runtime weight-file loader. Large vocabulary/rank values create large headers.

The exported bank predicts raw offset-indexed token guesses. For verification,
prepend the target's offset1 token and use a contiguous ordered range of learned
offsets. The current C core exposes borrowed immutable arrays; integration with
GPU kernels, weight artifacts and engine discovery remains pending.

The synthetic unit test proves training/export alignment, not useful real-model
acceptance. Training loss alone is not validation. Save a disjoint held-out corpus,
conditional acceptance at every offset and end-to-end bank/verify cost. Exported
packs mark `quality_validated: false`; changing that flag cannot certify a pack.

Future learned heads, prefetch buffers, hot-expert policy and INT8 KV are currently
experimental interfaces; `coli chat` has no new feature switch for them.

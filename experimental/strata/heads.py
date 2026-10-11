#!/usr/bin/env python3
"""Train independent low-rank future-token heads from frozen target hidden traces.

No model is downloaded. Traces must come from the exact intended checkpoint,
tokenizer, quantization and hidden-state tap. These heads only propose tokens.
Requires torch for training; trace validation and JSON export use the stdlib.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "c"))
from family_registry import FAMILIES

TRACE_SCHEMA = "colibri.strata.hidden-trace/v1"
PACK_SCHEMA = "colibri.strata.parallel-heads/v1"
MAX_FLOATS = 16_000_000
MAX_TRACE_BYTES = 256 * 1024 * 1024
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value, name, upper):
    require(type(value) is int and 1 <= value <= upper, f"invalid {name}")
    return value


def identity(meta):
    require(isinstance(meta, dict), "metadata must be an object")
    text_families = {f.id for f in FAMILIES if f.modality == "text"}
    require(isinstance(meta.get("family"), str) and meta["family"] in text_families,
            "family must be an upstream text family")
    require(isinstance(meta.get("checkpoint_revision"), str)
            and bool(re.fullmatch(r"[0-9a-f]{40}", meta["checkpoint_revision"])),
            "checkpoint_revision must be a pinned 40-digit commit")
    require(isinstance(meta.get("tokenizer_sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", meta["tokenizer_sha256"])),
            "tokenizer_sha256 must be 64 lowercase hex digits")
    for key in ("hidden_tap", "quantization"):
        require(isinstance(meta.get(key), str) and meta[key].strip()
                and len(meta[key]) <= 1024 and "\0" not in meta[key], f"invalid {key}")
    positive_int(meta.get("hidden_size"), "hidden_size", 32768)
    positive_int(meta.get("vocab_size"), "vocab_size", 1048576)
    result = {key: meta[key] for key in ("family", "checkpoint_revision", "tokenizer_sha256",
                                      "hidden_tap", "quantization", "hidden_size", "vocab_size")}
    if "native_fingerprint" in meta:
        require(isinstance(meta["native_fingerprint"],str) and
                re.fullmatch(r"[0-9a-f]{64}",meta["native_fingerprint"]), "invalid native target fingerprint")
        result["native_fingerprint"] = meta["native_fingerprint"]
    return result


def load_trace(path, offsets):
    require(offsets and len(offsets) <= 8 and offsets == sorted(set(offsets))
            and all(type(n) is int and 2 <= n <= 9 for n in offsets),
            "offsets must be unique increasing integers in 2..9; offset1 is the target's pick")
    path = Path(path)
    require(path.stat().st_size <= MAX_TRACE_BYTES, "trace exceeds 256 MiB limit")
    hidden, targets = [], []
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream, "{}"))
        require(isinstance(header, dict) and header.get("schema") == TRACE_SCHEMA, "unsupported trace schema")
        meta = identity(header)
        width, vocab = meta["hidden_size"], meta["vocab_size"]
        for number, line in enumerate(stream, 2):
            row = json.loads(line)
            require(isinstance(row, dict), f"line {number}: row must be an object")
            h, future = row.get("hidden"), row.get("future_tokens")
            require(isinstance(h, list) and len(h) == width,
                    f"line {number}: hidden width differs from metadata")
            require(all(type(x) in (int, float) and abs(x) <= FLOAT32_MAX and math.isfinite(x) for x in h),
                    f"line {number}: nonfinite or unrepresentable hidden value")
            require(isinstance(future, list), f"line {number}: missing future_tokens")
            require(all(x is None or (type(x) is int and 0 <= x < vocab) for x in future),
                    f"line {number}: invalid future token")
            labels = [future[o - 1] if o <= len(future) else None for o in offsets]
            if all(x is None for x in labels):
                continue
            require((len(hidden) + 1) * width <= MAX_FLOATS, "hidden trace exceeds float limit")
            hidden.append(h)
            targets.append([-100 if x is None else x for x in labels])
    require(hidden, "trace contains no trainable rows")
    require(all(any(row[k] >= 0 for row in targets) for k in range(len(offsets))),
            "every requested offset needs at least one valid training label")
    return meta, hidden, targets


def _feature_scale(x):
    """Fixed training-only RMS scaling, folded into down weights on export.

    Double statistics avoid overflow from finite F32 outliers. A zero/tiny
    coordinate uses the floor rather than amplifying noise without a bound.
    No centering or per-example normalization is needed by the native runner.
    """
    import torch
    rms = x.to(torch.float64).square().mean(dim=0).sqrt()
    scale = torch.where(rms >= 1e-8, rms.clamp_min(1e-8).reciprocal(), torch.ones_like(rms)).to(torch.float32)
    require(bool(torch.all(torch.isfinite(scale))), "nonfinite feature scale")
    return scale


def _shuffled_batches(size, batch_size, generator):
    import torch
    order, cursor = torch.randperm(size, generator=generator), 0
    while True:
        pieces, remaining = [], batch_size
        while remaining:
            take = min(remaining, size - cursor)
            pieces.append(order[cursor:cursor + take])
            cursor += take
            remaining -= take
            if cursor == size:
                order, cursor = torch.randperm(size, generator=generator), 0
        yield torch.cat(pieces)


def train(path, offsets, rank, steps, batch_size, learning_rate, seed, *, progress=None):
    positive_int(rank, "rank", 1024)
    positive_int(steps, "steps", 100000)
    positive_int(batch_size, "batch_size", 4096)
    require(type(learning_rate) in (int, float) and 0 < learning_rate <= 1
            and math.isfinite(learning_rate), "learning rate must be in (0,1]")
    require(type(seed) is int and 0 <= seed < 2**63, "invalid seed")
    meta, h, labels = load_trace(path, offsets)
    count, width, vocab = len(offsets), meta["hidden_size"], meta["vocab_size"]
    require(count * (rank * (width + vocab) + vocab) <= MAX_FLOATS,
            "head weights exceed float limit; reduce rank or head count")
    require(count * batch_size * vocab <= MAX_FLOATS,
            "training logits exceed float limit; reduce batch size")
    import torch
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    x = torch.tensor(h, dtype=torch.float32)
    scale = _feature_scale(x)
    x *= scale
    require(bool(torch.all(torch.isfinite(x))), "nonfinite normalized feature")
    y = torch.tensor(labels, dtype=torch.long).T.contiguous()
    down = torch.nn.Parameter(torch.randn(count, rank, width, generator=generator) / math.sqrt(width))
    up = torch.nn.Parameter(torch.randn(count, vocab, rank, generator=generator) * 0.01)
    # A unit total pseudocount avoids spending most updates suppressing the
    # hundreds of thousands of classes absent from a small trace. Counts and
    # feature statistics are derived only from the supplied training rows.
    priors = torch.stack([torch.bincount(row[row >= 0], minlength=vocab).float() + 1. / vocab for row in y])
    bias = torch.nn.Parameter(priors.log())
    optimizer = torch.optim.AdamW([{"params": [down], "lr": learning_rate * 0.1},
                                   {"params": [up, bias], "lr": learning_rate}], weight_decay=0.01)
    warmup_steps = min(100, max(1, steps // 10))
    batches = _shuffled_batches(len(x), batch_size, generator)
    losses = []
    for step in range(steps):
        # Seeded epoch shuffles prevent adjacent-session minibatches from
        # repeatedly overwriting the previous session's updates. Every offset
        # still reads the SAME x; no
        # earlier head's prediction or hidden state feeds a later head.
        indices = next(batches)
        warmup = min(1., (step + 1) / warmup_steps)
        decay = 0.1 + 0.9 * 0.5 * (1. + math.cos(math.pi * step / max(1, steps - 1)))
        optimizer.param_groups[0]["lr"] = learning_rate * 0.1 * warmup * decay
        optimizer.param_groups[1]["lr"] = learning_rate * warmup * decay
        latent = torch.einsum("bh,krh->kbr", x[indices], down)
        logits = torch.einsum("kbr,kvr->kbv", latent, up) + bias[:, None, :]
        selected = y[:, indices]
        # One CE over all valid head labels also handles a batch in which one
        # offset has no valid labels without producing NaN.
        if not torch.any(selected >= 0):
            continue
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), selected.reshape(-1))
        require(bool(torch.isfinite(loss)), "training produced nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_((down, up, bias), 1.)
        require(bool(torch.isfinite(norm)), "training produced nonfinite gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        if progress is not None and (step == 0 or (step + 1) % 100 == 0 or step + 1 == steps):
            progress({"step": step + 1, "loss": losses[-1], "gradient_norm": float(norm),
                      "learning_rate": optimizer.param_groups[1]["lr"]})
    require(losses, "no trainable minibatch")
    heads = [{"future_offset": offset, "rank": rank,
              "down": (down[k].detach() * scale).flatten().tolist(),
              "up": up[k].detach().flatten().tolist(),
              "bias": bias[k].detach().tolist()} for k, offset in enumerate(offsets)]
    require(all(math.isfinite(v) for head in heads for key in ("down", "up", "bias") for v in head[key]),
            "nonfinite exported weight")
    return {"schema": PACK_SCHEMA, "identity": meta, "heads": heads,
            "training": {"trace_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         "seed": seed, "steps": steps, "batch_size": batch_size,
                         "learning_rate": learning_rate, "first_loss": losses[0],
                         "last_loss": losses[-1], "quality_validated": False,
                         "feature_normalization": "coordinate-rms-folded-into-down/v1",
                         "feature_scale_min": float(scale.min()), "feature_scale_max": float(scale.max()),
                         "optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip_norm": 1.,
                         "down_learning_rate_multiplier": 0.1, "warmup_steps": warmup_steps,
                         "schedule": "cosine-to-0.1", "minibatches": "seeded-epoch-shuffle"}}


def validate_pack(pack, expected_identity=None):
    require(isinstance(pack, dict) and pack.get("schema") == PACK_SCHEMA, "unsupported head pack")
    meta = identity(pack.get("identity"))
    if expected_identity is not None:
        require(meta == identity(expected_identity), "head pack belongs to a different target/tap/tokenizer")
    bank = pack.get("heads")
    require(isinstance(bank, list) and 1 <= len(bank) <= 8, "head count must be in 1..8")
    offsets, total = [], 0
    for head in bank:
        require(isinstance(head, dict), "head must be an object")
        offset = positive_int(head.get("future_offset"), "future_offset", 9)
        require(offset >= 2, "offset1 is the target's own selection")
        rank = positive_int(head.get("rank"), "rank", 1024)
        offsets.append(offset)
        for key, length in (("down", rank * meta["hidden_size"]),
                            ("up", rank * meta["vocab_size"]), ("bias", meta["vocab_size"])):
            array = head.get(key)
            require(isinstance(array, list) and len(array) == length, f"wrong {key} dimensions")
            require(all(type(v) in (int, float) and abs(v) <= FLOAT32_MAX and math.isfinite(v) for v in array),
                    f"invalid {key} weight")
            total += length
    require(offsets == sorted(set(offsets)), "head offsets must be unique and increasing")
    require(total <= MAX_FLOATS, "head pack exceeds float limit")
    return meta


def c_string(value):
    # Fixed-width octal escapes encode UTF-8 bytes without surrogate UCNs,
    # hexadecimal escape bleed, or JSON's different escaping rules.
    return '"' + "".join(chr(b) if 32 <= b <= 126 and b not in (34, 92)
                         else f"\\{b:03o}" for b in value.encode("utf-8")) + '"'


def export_c(pack, prefix="strata_trained"):
    """Trusted build-time export; no runtime untrusted-weight file parser."""
    meta = validate_pack(pack)
    require(bool(re.fullmatch(r"[a-z][a-z0-9_]*", prefix)), "invalid C symbol prefix")
    lines = ["/* Generated experimental weights. Target verification is mandatory. */",
             '#include "strata_heads.h"']
    # A family adapter must compare these identities with the actual target
    # before attaching the bank. Equal shapes alone do not imply compatibility.
    for key in ("family", "checkpoint_revision", "tokenizer_sha256", "hidden_tap", "quantization"):
        lines.append(f"static const char {prefix}_{key}[] = {c_string(meta[key])};")
    for k, head in enumerate(pack["heads"]):
        for key in ("down", "up", "bias"):
            numbers = []
            for number in head[key]:
                literal = format(number, ".9g")
                if "." not in literal and "e" not in literal.lower():
                    literal += ".0"
                numbers.append(literal + "f")
            lines.append(f"static const float {prefix}_{k}_{key}[] = {{")
            lines.extend("    " + ", ".join(numbers[i:i + 8]) + "," for i in range(0, len(numbers), 8))
            lines.append("};")
    lines.append(f"static const strata_heads_head {prefix}_heads[] = {{")
    for k, head in enumerate(pack["heads"]):
        lines.append("    {" + f".future_offset={head['future_offset']}, .rank={head['rank']}, "
                     f".down={prefix}_{k}_down, .down_count={len(head['down'])}, "
                     f".up={prefix}_{k}_up, .up_count={len(head['up'])}, "
                     f".bias={prefix}_{k}_bias, .bias_count={len(head['bias'])}" + "},")
    lines.append("};")
    lines.append(f"static const strata_heads_bank {prefix}_bank = {{"
                 f".hidden_dim={meta['hidden_size']}, .vocab_size={meta['vocab_size']}, "
                 f".head_count={len(pack['heads'])}, .heads={prefix}_heads" + "};")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--export-c", help="also export a trusted build-time C header")
    parser.add_argument("--weight-bits", type=int, choices=(4, 8, 16, 32), default=32,
                        help="C export storage precision; INT16 is integer, not FP16")
    parser.add_argument("--block-values", type=int, default=64)
    parser.add_argument("--offsets", default="2,3,4")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    try:
        offsets = [int(x) for x in args.offsets.split(",")]
        pack = train(args.trace, offsets, args.rank, args.steps, args.batch_size,
                     args.learning_rate, args.seed)
        output = Path(args.out)
        require(not output.exists(), "output already exists; use a new filename")
        require(output.resolve() != Path(args.trace).resolve(), "output cannot replace trace")
        if args.export_c and args.weight_bits != 32:
            import precision
            c_text = precision.export_c(pack, args.weight_bits, args.block_values)
        else:
            c_text = export_c(pack) if args.export_c else None
        if args.export_c:
            require(not Path(args.export_c).exists(), "C output already exists")
            require(Path(args.export_c).resolve() != output.resolve(), "C and JSON outputs must differ")
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(pack, allow_nan=False) + "\n")
        if c_text:
            with Path(args.export_c).open("x", encoding="utf-8") as stream:
                stream.write(c_text)
        print(json.dumps({"output": str(output), "heads": len(pack["heads"]),
                          "training": pack["training"]}))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

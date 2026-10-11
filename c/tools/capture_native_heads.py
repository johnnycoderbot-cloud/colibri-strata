#!/usr/bin/env python3
"""Capture frozen native target features and labels for independent token heads.

The first qualified adapter is qwen36 CPU INT8/integer-dot. All future labels
come from the target's own subsequent selections. Session boundaries and
positions remain in the trace so validation can hold out whole requests.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from export_native_heads import fingerprint_info
from native_decode_bench import NativeEngine, linux_path

PRECISION = "qwen36:DBITS=8;IDOT=1;kernel=cpu;tap=final-rmsnorm+next-embedding/v1"
MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_FLOATS = 16_000_000


def convert_raw(raw, output, revision, expected):
    if raw.stat().st_size > MAX_RAW_BYTES:
        raise ValueError("raw trace exceeds 256 MiB")
    groups = {}
    with raw.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (header.get("schema") != "colibri.native-hidden-raw/v1" or
                header.get("native_fingerprint") != expected["fingerprint"] or
                header.get("hidden_size") != expected["hidden_size"] or
                header.get("vocab_size") != expected["vocab_size"] or
                header.get("quantization") != PRECISION or
                header.get("hidden_tap") != "final-rmsnorm+next-embedding/v1"):
            raise ValueError("native capture identity differs from requested target")
        for line in stream:
            row = json.loads(line)
            session, position = row["session"], row["position"]
            if type(session) is not int or session < 1 or type(position) is not int or position < 0:
                raise ValueError("invalid native session/position")
            rows = groups.setdefault(session, [])
            if rows and position != rows[-1]["position"] + 1:
                raise ValueError("capture contains a discontinuous session")
            if (not isinstance(row["hidden"], list) or
                    len(row["hidden"]) != expected["hidden_size"] or
                    not all(type(x) in (int, float) and math.isfinite(x) and
                            abs(x) <= float.fromhex("0x1.fffffep+127") for x in row["hidden"]) or
                    type(row["next_token"]) is not int or
                    not 0 <= row["next_token"] < expected["vocab_size"]):
                raise ValueError("invalid native feature/label")
            rows.append(row)
    meta = {"schema": "colibri.strata.hidden-trace/v1", "family": "qwen36",
            "checkpoint_revision": revision, "tokenizer_sha256": expected["tokenizer_sha256"],
            "hidden_tap": header["hidden_tap"], "quantization": PRECISION,
            "hidden_size": header["hidden_size"], "vocab_size": header["vocab_size"],
            "native_fingerprint": header["native_fingerprint"]}
    count = sum(max(0, len(rows) - 1) for rows in groups.values())
    if not count or count * meta["hidden_size"] > MAX_FLOATS:
        raise ValueError("capture is empty or exceeds training float budget")
    # Write only after the identity, positions and dimensions have been checked.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(meta, allow_nan=False) + "\n")
        for session, rows in groups.items():
            for index, row in enumerate(rows[:-1]):
                future = [r["next_token"] for r in rows[index:index + 9]]
                stream.write(json.dumps({"session": session, "position": row["position"],
                                         "hidden": row["hidden"], "future_tokens": future},
                                        allow_nan=False) + "\n")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="pinned source checkpoint commit")
    parser.add_argument("--prompts", required=True, help="JSON array of request strings")
    parser.add_argument("--raw", required=True, help="new native JSONL output")
    parser.add_argument("--output", required=True, help="new training JSONL output")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--wsl")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a pinned 40-digit commit")
    if not 4 <= args.tokens <= 512 or not 1 <= args.threads <= 64:
        parser.error("tokens must be 4..512 and threads 1..64")
    raw, output = Path(args.raw).resolve(), Path(args.output).resolve()
    protected = {Path(args.prompts).resolve(), Path(args.engine).resolve()}
    protected.update(p.resolve() for p in Path(args.model).glob("*"))
    if raw == output or raw in protected or output in protected or raw.exists() or output.exists():
        parser.error("raw and output must be distinct new files outside model assets")
    if Path(args.prompts).stat().st_size > 8 * 1024 * 1024:
        parser.error("prompt file exceeds 8 MiB")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 128 or
            not all(isinstance(p, str) and 0 < len(p.encode("utf-8")) <= 65536 for p in prompts)):
        parser.error("prompts must contain 1..128 nonempty request strings of at most 64 KiB")
    expected = fingerprint_info(args.model, PRECISION)
    if expected.get("family") != "qwen36":
        parser.error("capture adapter currently requires qwen36")
    if expected.get("checkpoint_revision", args.revision) != args.revision:
        parser.error("revision differs from native config")
    if len(prompts) * args.tokens * expected["hidden_size"] > MAX_FLOATS:
        parser.error("requested capture exceeds training float budget")
    env = {"COLI_STRATA_TRACE": linux_path(raw) if args.wsl else str(raw),
           "COLI_STRATA_INPUT": "next-embedding"}
    engine = NativeEngine(args.engine, args.model, args.threads, env, args.wsl)
    try:
        for session, prompt in enumerate(prompts, 1):
            result = engine.turn(prompt, args.tokens)
            if raw.stat().st_size > MAX_RAW_BYTES:
                raise ValueError("capture exceeds raw byte budget")
            print(json.dumps({"session": session, "tokens": result["tokens"],
                              "output_sha256": result["output_sha256"]}), flush=True)
    finally:
        engine.close()
    rows = convert_raw(raw, output, args.revision, expected)
    print(json.dumps({"rows": rows, "sessions": len(prompts),
                      "native_fingerprint": expected["fingerprint"],
                      "trace_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

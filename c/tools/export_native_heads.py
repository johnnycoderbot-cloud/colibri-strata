#!/usr/bin/env python3
"""Export existing trained JSON heads to bounded, fingerprinted CSNH v1.

No weights are trained or fabricated here. True signed CSQ1 INT4/8/16 source
storage is decoded once by the native loader for F32 CPU proposal computation.
The model fingerprint binds actual native model/tokenizer bytes and the exact
runtime precision descriptor. Payload SHA256 detects corruption, not authorship.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experimental" / "strata"))
import heads
import precision

MAX_FILE_BYTES = 512 * 1024 * 1024
MODEL_TAG = b"CSNH-model-v1\0"
AUGMENTED_TAP = "final-rmsnorm+next-embedding/v1"


def _augmented_tap(runtime_precision):
    # The native engine appends this exact terminal field. Do not accidentally
    # interpret a substring in an opaque legacy precision descriptor as a tap.
    fields = runtime_precision.split(":", 1)[-1].split(";")
    marker = "tap=" + AUGMENTED_TAP
    if marker not in fields:
        return False
    heads.require(fields[-1] == marker and sum(field.startswith("tap=") for field in fields) == 1,
                  "augmented tap must be the single terminal tap field")
    return True


def _json(path, limit=MAX_FILE_BYTES):
    heads.require(path.stat().st_size <= limit, "JSON exceeds byte limit")
    def pairs(values):
        result = {}
        for key, value in values:
            heads.require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)


def _model_files(model_dir):
    root = Path(model_dir)
    heads.require(root.is_dir(), "native model directory missing")
    names = [path for path in root.iterdir() if path.is_file() and
             (path.name in ("config.json", "tokenizer.json", "qwen36_meta.json") or
              len(path.name) > 12 and path.name.endswith(".safetensors"))]
    names.sort(key=lambda path: path.name.encode("utf-8"))
    labels = {path.name for path in names}
    heads.require("config.json" in labels and "tokenizer.json" in labels and
                  any(name.endswith(".safetensors") for name in labels), "native model needs config, tokenizer and safetensors")
    heads.require(len(names) <= 4096 and all(len(path.name.encode("utf-8")) <= 4096 for path in names), "model file labels exceed bounds")
    return names


def fingerprint_model(model_dir, runtime_precision):
    heads.require(isinstance(runtime_precision, str) and runtime_precision and "\0" not in runtime_precision,
                  "runtime precision must be an exact nonempty descriptor")
    encoded_precision = runtime_precision.encode("utf-8")
    heads.require(len(encoded_precision) <= 1024, "runtime precision descriptor exceeds bounds")
    files = _model_files(model_dir)
    digest = hashlib.sha256(MODEL_TAG)
    digest.update(struct.pack("<Q", len(encoded_precision)))
    digest.update(encoded_precision)
    digest.update(struct.pack("<Q", len(files)))
    for path in files:
        label = path.name.encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack("<Q", len(label)))
        digest.update(label)
        digest.update(struct.pack("<Q", size))
        read = 0
        with path.open("rb") as stream:
            while chunk := stream.read(65536):
                digest.update(chunk)
                read += len(chunk)
        heads.require(read == size, "model file changed during fingerprinting")
    return digest.digest()


def fingerprint_info(model_dir, runtime_precision):
    root = Path(model_dir)
    config = _json(root / "config.json", 1024 * 1024)
    heads.require(isinstance(config, dict), "native config must be an object")
    text = config.get("text_config", config)
    heads.require(isinstance(text, dict), "native text config must be an object")
    hidden = heads.positive_int(text.get("hidden_size"), "hidden_size", 32768)
    vocab = heads.positive_int(text.get("vocab_size"), "vocab_size", 1048576)
    fingerprint = fingerprint_model(root, runtime_precision).hex()
    augmented = _augmented_tap(runtime_precision)
    feature_width = heads.positive_int(hidden * (2 if augmented else 1), "feature hidden_size", 32768)
    # Use the authoritative registry when the config identifies a known text
    # family; old exports without model_type retain their existing behavior.
    types = {model_type: family.id for family in heads.FAMILIES if family.modality == "text"
             for model_type in family.model_types}
    families = {types[model_type.strip().lower()] for source in (config, text)
                if isinstance(model_type := source.get("model_type"), str)
                and model_type.strip().lower() in types}
    precision_family = runtime_precision.split(":", 1)[0]
    if precision_family in {family.id for family in heads.FAMILIES if family.modality == "text"}:
        families.add(precision_family)
    heads.require(len(families) <= 1, "native config and runtime precision families differ")
    revisions = {revision for source in (config, text)
                 if isinstance(revision := source.get("_commit_hash"), str)
                 and re.fullmatch(r"[0-9a-f]{40}", revision)}
    heads.require(len(revisions) <= 1, "native config checkpoint revisions differ")
    tokenizer = hashlib.sha256()
    with (root / "tokenizer.json").open("rb") as stream:
        while chunk := stream.read(65536):
            tokenizer.update(chunk)
    result = {"fingerprint": fingerprint, "hidden_size": feature_width, "vocab_size": vocab,
              "runtime_precision": runtime_precision, "tokenizer_sha256": tokenizer.hexdigest()}
    if augmented:
        # These are features from h_t and the target's already-selected token1,
        # not future target states and not autoregressive head-to-head inputs.
        result.update(native_hidden_size=hidden, hidden_tap=AUGMENTED_TAP)
    if families:
        result["family"] = next(iter(families))
    if revisions:
        result["checkpoint_revision"] = next(iter(revisions))
    return result


def export_pack(pack, model_dir, runtime_precision, bits=4, block_values=64):
    identity = heads.validate_pack(pack)
    native = fingerprint_info(model_dir, runtime_precision)
    heads.require(identity.get("native_fingerprint") == native["fingerprint"],
                  "head native_fingerprint must match actual native model bytes, precision and tap")
    heads.require(identity["hidden_size"] == native["hidden_size"] and identity["vocab_size"] == native["vocab_size"], "native model/head dimensions differ")
    heads.require(identity["tokenizer_sha256"] == native["tokenizer_sha256"], "head tokenizer differs from native model")
    heads.require(identity["quantization"] == runtime_precision, "head training precision differs from native runtime precision")
    if "family" in native:
        heads.require(identity["family"] == native["family"], "head family differs from native model")
    if "checkpoint_revision" in native:
        heads.require(identity["checkpoint_revision"] == native["checkpoint_revision"], "head checkpoint differs from native config revision")
    if "hidden_tap" in native:
        heads.require(identity["hidden_tap"] == native["hidden_tap"], "head hidden tap differs from augmented native tap")
    else:
        heads.require(identity["hidden_tap"] != AUGMENTED_TAP, "augmented head tap requires augmented runtime precision")
    heads.require([head["future_offset"] for head in pack["heads"]] == list(range(2, len(pack["heads"]) + 2)), "native head offsets must be contiguous from 2")
    payload = bytearray()
    for head in pack["heads"]:
        down = precision.encode(head["down"], bits, block_values)
        up = precision.encode(head["up"], bits, block_values)
        bias = b"".join(struct.pack("<f", precision.f32(v)) for v in head["bias"])
        payload.extend(struct.pack("<IIQQII", head["future_offset"], head["rank"], len(down), len(up), len(head["bias"]), 0))
        payload.extend(down); payload.extend(up); payload.extend(bias)
        heads.require(96 + len(payload) <= MAX_FILE_BYTES, "native head pack exceeds 512 MiB")
    header = struct.pack("<4s7I", b"CSNH", 1, 96, identity["hidden_size"], identity["vocab_size"], len(pack["heads"]), bits, 0)
    return header + bytes.fromhex(native["fingerprint"]) + hashlib.sha256(payload).digest() + bytes(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-precision", required=True)
    parser.add_argument("--fingerprint-only", action="store_true")
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--block-values", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.fingerprint_only:
            print(json.dumps(fingerprint_info(args.model_dir, args.runtime_precision), sort_keys=True))
            return 0
        heads.require(args.pack is not None and args.output is not None, "--pack and --output are required for export")
        output = args.output.resolve()
        protected = {path.resolve() for path in _model_files(args.model_dir)} | {args.pack.resolve()}
        heads.require(output not in protected, "output must not overwrite model assets or its source pack")
        blob = export_pack(_json(args.pack), args.model_dir, args.runtime_precision, args.bits, args.block_values)
        with output.open("xb") as stream:
            stream.write(blob)
        print(json.dumps({"output": str(args.output), "bytes": len(blob), "source_bits": args.bits,
                          "compute": "F32 CPU proposals", "fingerprint": blob[32:64].hex()}))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Active-bank native Qwen correctness gates on a caller-supplied checkpoint.

The constant, biased CSNH banks are SYNTHETIC CORRECTNESS FIXTURES ONLY. They
deliberately use a plain-target oracle to force acceptance/rejection paths;
their results are never evidence of trained-head accuracy or acceleration.
No checkpoint download, capture device, or accelerator is opened here.

Example from the repository root (Windows hosting a CPU engine in WSL):
  .venv-hardware/Scripts/python.exe c/tests/native_head_frontier_harness.py \
      --engine c/qwen36 --model models-native/qwen35-08 --wsl kali-pc \
      --work outputs-native/frontier-gates --output outputs-native/frontier.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "c" / "tools"))
from native_decode_bench import NativeEngine, READY, linux_path  # noqa: E402
import export_native_heads as exporter  # noqa: E402

PRECISION = "qwen36:DBITS=8;IDOT=1;kernel=cpu;tap=final-rmsnorm+next-embedding/v1"
PROMPT = ("<|im_start|>user\nExplain binary search in complete sentences, "
          "with a concrete example.<|im_end|>\n<|im_start|>assistant\n"
          "<think>\n\n</think>\n\n")
EOS_PROMPT = ("<|im_start|>user\nAnswer with exactly these three words and no punctuation: "
              "red green blue<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
CONTINUATION = (b"<|im_end|>\n<|im_start|>user\nExplain the time complexity of binary search."
                b"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
STATS = re.compile(r"\[qwen36 strata\] turn (\S+?)(?: cancelled)?: (.*)")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


class Driver:
    """Extended requests over the existing bounded asynchronous frame reader."""
    def __init__(self, args, label, bank=None, capture=True):
        self.args, self.label, self.count = args, label, 0
        self.trace = args.work / (label + ".jsonl") if capture else None
        env = {"COLI_PREFIX_LOG": "1", "COLI_SPEC_GATE": "0",
               "COLI_STRATA_INPUT": "next-embedding", "COLI_LOOKUP": "0",
               "COLI_CUDA": "0", "COLI_VULKAN": "0"}
        if bank:
            env["COLI_STRATA_HEADS"] = linux_path(bank) if args.wsl else str(bank)
        if self.trace:
            env["COLI_STRATA_TRACE"] = linux_path(self.trace) if args.wsl else str(self.trace)
        self.engine = NativeEngine(args.engine, args.model, args.threads, env, args.wsl)
        self.log = ""

    def ask(self, prompt, tokens=10, pin=False, temperature=0, cancel_after=None):
        self.count += 1
        rid = "frontier" + str(self.count)
        payload = prompt.encode("utf-8") if isinstance(prompt, str) else prompt
        extension = " logprobs=2" + (" pin=1" if pin else "")
        header = f"SUBMIT {rid} 0 {len(payload)} {tokens} {temperature} 0.95{extension}\n"
        self.engine.process.stdin.write(header.encode() + payload + b"\n")
        self.engine.process.stdin.flush()
        deadline = time.monotonic() + self.args.request_timeout
        frames, bodies, ids, cancelled = [], [], [], False
        while True:
            try:
                line, body = self.engine.frames.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty as error:
                raise TimeoutError(f"{self.label}/{rid}: request timed out") from error
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.label}/{rid}: request timed out")
            require(line, f"{self.label}/{rid}: engine exited: {''.join(self.engine.stderr)[-2000:]}")
            fields = line.split()
            if len(fields) < 2 or fields[1] != rid.encode():
                continue
            if fields[0] == b"DATA":
                require(len(fields) >= 7 and body is not None, "missing token/logprob evidence")
                ids.append(int(fields[5]))
                bodies.append(body)
                # Compare exact numeric DATA content, excluding only request id.
                frames.append(b" ".join(fields[2:]) + b"|" + body)
                if cancel_after and len(ids) >= cancel_after and not cancelled:
                    self.engine.process.stdin.write(f"CANCEL {rid}\n".encode())
                    self.engine.process.stdin.flush()
                    cancelled = True
            elif fields[0] == b"ERROR":
                require(cancelled and fields[2:] == [b"CANCELLED"], line.decode("utf-8", "replace"))
                return {"id": rid, "ids": ids, "frames": frames, "body": b"".join(bodies), "cancelled": True}
            elif fields[0] == b"DONE":
                require(not cancelled, "cancellation arrived after the request finished")
                require(fields[2] == b"STAT" and int(fields[3]) == len(ids), "bad DONE token accounting")
                return {"id": rid, "ids": ids, "frames": frames, "body": b"".join(bodies), "cancelled": False}

    def close(self):
        self.engine.close()
        self.log = "".join(self.engine.stderr)
        with (self.args.work / (self.label + ".log")).open("x", encoding="utf-8") as stream:
            stream.write(self.log)


def compare(a, b, prefix=False):
    expected = a["frames"][:len(b["frames"])] if prefix else a["frames"]
    require(expected == b["frames"], f"numeric DATA/body mismatch: {a['ids']} vs {b['ids']}")
    require(a["ids"][:len(b["ids"])] == b["ids"] if prefix else a["ids"] == b["ids"], "token IDs differ")


def read_trace(path):
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        # Keep the serialized F32-round-trippable values too: JSON numeric
        # equality alone would silently identify signed -0 with +0.
        rows = [(json.loads(line), line.rstrip()) for line in stream]
    require(header["schema"] == "colibri.native-hidden-raw/v1", "missing native trace identity")
    return header, rows


def compare_trace(gold, feature, first_session_only=False, prefix=False):
    gh, gr = read_trace(gold.trace)
    fh, fr = read_trace(feature.trace)
    require(gh == fh, "target/feature trace fingerprints differ")
    if first_session_only:
        gr = [r for r in gr if r[0]["session"] == 1]
        fr = [r for r in fr if r[0]["session"] == 1]
    if prefix:
        gr = gr[:len(fr)]
    require(gr == fr, "accepted/rejected hidden frontier differs from target-only trace")
    require(fr, "no live frontier was captured")
    return len(fr)


def stats(driver, rid="frontier1"):
    match = next((m for line in driver.log.splitlines() if (m := STATS.fullmatch(line)) and m[1] == rid), None)
    require(match is not None, f"no active-bank statistics for {rid}: {driver.log[-2000:]}")
    return {key: float(value) if "." in value else int(value)
            for key, value in re.findall(r"([a-z_]+)=([0-9.]+)", match[2])}


def ended_ahead(driver, rid="frontier1", cancelled=False):
    values = stats(driver, rid)
    if "ended_mid_verify" in values:
        return values["ended_mid_verify"] == 1
    scope = rid + (" cancelled" if cancelled else "") + ":"
    return any("[qwen36 lookup] turn " + scope in line and "ended mid-verify" in line
               for line in driver.log.splitlines())


def constant_bank(args, native, name, tokens):
    width, vocab = native["hidden_size"], native["vocab_size"]
    identity = {"family": "qwen36", "checkpoint_revision": args.revision,
                "tokenizer_sha256": native["tokenizer_sha256"], "native_fingerprint": native["fingerprint"],
                "hidden_tap": exporter.AUGMENTED_TAP, "quantization": PRECISION,
                "hidden_size": width, "vocab_size": vocab}
    pack = {"schema": exporter.heads.PACK_SCHEMA, "identity": identity, "heads": [],
            "training": {"synthetic_correctness_fixture_only": True, "quality_validated": False}}
    for index, token in enumerate(tokens):
        bias = [0.0] * vocab
        bias[token] = 1.0
        pack["heads"].append({"future_offset": index + 2, "rank": 1,
                              "down": [0.0] * width, "up": [0.0] * vocab, "bias": bias})
    path = args.work / (name + ".snh")
    with path.open("xb") as stream:
        stream.write(exporter.export_pack(pack, args.model, PRECISION, bits=4))
    return path


def paired_case(args, name, bank, budget=10, continuation=False, pin=False, sample=False, prompt=PROMPT):
    gold = feature = None
    try:
        gold = Driver(args, name + "-plain")
        feature = Driver(args, name + "-heads", bank)
        a = gold.ask(prompt, budget, pin, 0.8 if sample else 0)
        b = feature.ask(prompt, budget, pin, 0.8 if sample else 0)
        compare(a, b)
        if continuation:
            extended = prompt.encode() + a["body"] + CONTINUATION
            continued = gold.ask(extended, 8)
            require(len(continued["ids"]) >= 4, "continuation ended before state was meaningfully exercised")
            compare(continued, feature.ask(extended, 8))
        if pin:
            # Diverge behind the pinned prompt; live-prefix reuse cannot perform this rewind.
            branch = PROMPT + " Give a different practical example."
            compare(gold.ask(branch, 8), feature.ask(branch, 8))
    finally:
        if gold:
            gold.close()
        if feature:
            feature.close()
    rows = compare_trace(gold, feature)
    if sample:
        require("[strata] loaded" in feature.log, "sample fallback never loaded the test bank")
        require("[qwen36 strata]" not in feature.log, "sampled request executed unqualified head path")
        return {"name": name, "tokens": len(b["ids"]), "frontier_rows": rows, "sample_fallback": True}
    values = stats(feature)
    require(values["bank_calls"] > 0 and values["failures"] == 0, "bank was inactive or failed")
    if name == "first-reject":
        require(values["accepted"] == 0, "first-rejection fixture unexpectedly accepted a draft")
    if name == "middle-reject":
        require(values["accepted"] >= 1 and values["drafts"] > values["accepted"], "middle rejection was not reached")
    if name == "full-accept-next-proposal":
        require(values["accepted"] >= 2 and values["bank_calls"] >= 2, "full acceptance/new proposal was not reached")
    if name == "budget-clipped-verify":
        require(values["drafts"] == 1 and values["accepted"] == 1 and
                values["forwards"] == 2 and values["tokens"] == 3,
                "budget did not clip a five-head bank to exactly one consumable draft")
        require(not ended_ahead(feature), "fixed output budget speculated into the final unfed token")
    if name == "eos-mid-verify":
        require(2 <= len(b["ids"]) < budget and ended_ahead(feature),
                "EOS did not end generation with pending verifier rows")
    if continuation:
        require("[PREFIX] reusing" in feature.log, "continuation did not exercise live prefix reuse")
    if pin:
        require("[PIN] scatto" in feature.log and "[PIN] restoring" in feature.log, "pin restore path was not observed")
    return {"name": name, "tokens": len(b["ids"]), "frontier_rows": rows, "stats": values}


def cancellation_case(args, bank):
    gold = feature = None
    try:
        gold = Driver(args, "cancel-plain")
        feature = Driver(args, "cancel-heads", bank)
        a = gold.ask(PROMPT, 12)
        b = feature.ask(PROMPT, 12, cancel_after=3)
        require(b["cancelled"] and 3 <= len(b["ids"]) < 12, "cancellation was not observed during generation")
        compare(a, b, prefix=True)
        prompt = PROMPT.encode() + b["body"] + CONTINUATION
        continued = gold.ask(prompt, 8)
        require(len(continued["ids"]) >= 4, "post-cancel continuation ended too early")
        compare(continued, feature.ask(prompt, 8))
    finally:
        if gold:
            gold.close()
        if feature:
            feature.close()
    rows = compare_trace(gold, feature, first_session_only=True, prefix=True)
    _, gold_rows = read_trace(gold.trace)
    _, feature_rows = read_trace(feature.trace)
    gold_next = [row for row in gold_rows if row[0]["session"] == 2]
    feature_next = [row for row in feature_rows if row[0]["session"] == 2]
    require(gold_next == feature_next and len(feature_next) >= 2,
            "post-cancel hidden frontier was not restored exactly")
    values = stats(feature)
    require(values["verifies"] > 0, "cancellation preceded head verification")
    require(ended_ahead(feature, cancelled=True), "cancellation did not undo pending verifier rows")
    require("[PREFIX] reusing" in feature.log, "post-cancellation state was not reused")
    return {"name": "cancel-mid-verify", "emitted_tokens": len(b["ids"]), "frontier_rows": rows,
            "post_cancel_frontier_rows": len(feature_next), "stats": values}


def precision_refusals(args, bank):
    results = []
    for key, value in (("COLI_DENSE_IDOT", "0"), ("COLI_DENSE_BITS", "4"), ("COLI_DENSE_I8", "0")):
        env = {"SNAP": linux_path(args.model) if args.wsl else str(args.model), "SERVE": "1",
               "OMP_NUM_THREADS": str(args.threads), "COLI_NO_OMP_TUNE": "1", "COLI_LOOKUP": "0",
               "COLI_STRATA_INPUT": "next-embedding", "COLI_DENSE_IDOT": "1", "COLI_DENSE_BITS": "8",
               "COLI_DENSE_I8": "1", "COLI_STRATA_HEADS": linux_path(bank) if args.wsl else str(bank),
               "COLI_CUDA": "0", "COLI_VULKAN": "0", key: value}
        if args.wsl:
            command = ["wsl.exe", "-d", args.wsl, "--cd", linux_path(ROOT), "--", "env"]
            command += [k + "=" + v for k, v in env.items()] + [linux_path(args.engine), "1", "8"]
            child_env = None
        else:
            command = [str(args.engine), "1", "8"]
            child_env = dict(os.environ, **env)
        result = subprocess.run(command, env=child_env, input=b"", capture_output=True, timeout=args.request_timeout)
        require(result.returncode != 0 and READY not in result.stdout, "unqualified precision reached READY")
        require(b"initial native qualification requires" in result.stderr, "wrong precision failed for an unrelated reason")
        results.append({"name": "precision-refusal", "setting": key + "=" + value, "exit_code": result.returncode})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wsl")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--revision", default="0" * 40, help="optional pinned source revision; zero denotes synthetic fixture metadata")
    parser.add_argument("--work", type=Path, required=True, help="NEW directory for correctness-only banks and traces")
    parser.add_argument("--output", type=Path, required=True, help="NEW report path")
    parser.add_argument("--cases", help="optional comma-separated subset; reports only selected coverage")
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.request_timeout <= 600, "invalid execution bounds")
    require(re.fullmatch(r"[0-9a-f]{40}", args.revision), "revision must be 40 lowercase hex digits")
    args.engine, args.model, args.work, args.output = (p.resolve() for p in (args.engine, args.model, args.work, args.output))
    require(args.engine.is_file() and args.model.is_dir(), "native engine/model is absent")
    require(not args.work.exists() and not args.output.exists(), "work directory and report must be new")
    args.work.mkdir(parents=True)
    started = time.monotonic()
    oracle = Driver(args, "oracle", capture=False)
    try:
        truth = oracle.ask(PROMPT, 12)["ids"]
    finally:
        oracle.close()
    require(len(truth) >= 10, "oracle stopped too early for frontier gates")
    native = exporter.fingerprint_info(args.model, PRECISION)
    wrong = next(token for token in range(native["vocab_size"]) if token not in truth)
    reject = constant_bank(args, native, "constant-reject", [wrong])
    middle = constant_bank(args, native, "constant-middle", [truth[2], wrong])
    full = constant_bank(args, native, "constant-full", truth[2:4])
    deep = constant_bank(args, native, "constant-deep", truth[2:7])
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    eos_id = config.get("text_config", config).get("eos_token_id")
    require(type(eos_id) is int and 0 <= eos_id < native["vocab_size"], "model lacks a scalar EOS token")
    eos = constant_bank(args, native, "constant-eos", [eos_id])
    available_cases = {"first-reject", "middle-reject", "full-accept-next-proposal",
                       "budget-clipped-verify", "prefix-pin-restore", "sample-fallback",
                       "cancel-mid-verify", "precision-refusal", "eos-mid-verify"}
    selected = set(args.cases.split(",")) if args.cases else available_cases
    require(selected and selected <= available_cases, "unknown or empty case selection")
    records = []
    for name, bank, options in (
        ("first-reject", reject, {"continuation": True}),
        ("middle-reject", middle, {"continuation": True}),
        ("full-accept-next-proposal", full, {"continuation": True}),
        ("budget-clipped-verify", deep, {"budget": 4, "continuation": True}),
        ("prefix-pin-restore", full, {"continuation": True, "pin": True}),
        ("sample-fallback", full, {"sample": True}),
        ("eos-mid-verify", eos, {"budget": 20, "continuation": True, "prompt": EOS_PROMPT}),
    ):
        if name not in selected:
            continue
        print(json.dumps({"name": name, "status": "running"}), flush=True)
        record = paired_case(args, name, bank, **options)
        records.append(record)
        print(json.dumps(record), flush=True)
    if "cancel-mid-verify" in selected:
        record = cancellation_case(args, deep)
        records.append(record)
        print(json.dumps(record), flush=True)
    if "precision-refusal" in selected:
        records.extend(precision_refusals(args, full))
    report = {"schema": "colibri.native-head-frontier-correctness/v1", "verdict": "approved",
              "synthetic_correctness_fixtures_only": True, "performance_evidence": False,
              "engine_sha256": sha(args.engine), "native_fingerprint": native["fingerprint"],
              "runtime_precision": PRECISION, "oracle_ids": truth, "records": records,
              "selected_cases": sorted(selected),
              "elapsed_seconds": time.monotonic() - started}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("native head frontier: active-bank rollback, hidden traces, protocol and numeric-mode gates passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

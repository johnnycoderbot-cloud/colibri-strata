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
import struct
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
MTP_STATS = re.compile(r"\[qwen36 MTP\] turn (\S+?)(?: cancelled)?: (.*)")
RESIDUAL_SCHEMA = "colibri.strata.logit-residual-heads/v1"
RESIDUAL_FIXTURE_BIAS = 1e6
CASE_NAMES = ("first-reject", "middle-reject", "full-accept-next-proposal",
              "budget-clipped-verify", "prefix-pin-restore", "sample-fallback",
              "cancel-mid-verify", "precision-refusal", "eos-mid-verify")
MTP_CASE_NAMES = CASE_NAMES + ("cycle-rejections", "actual-depth1", "actual-depth2", "actual-depth3")


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
    def __init__(self, args, label, bank=None, capture=True, *, mtp=False, force="", oracle=None, depth=None):
        self.args, self.label, self.count = args, label, 0
        self.mtp_active = mtp
        self.trace = args.work / (label + ".jsonl") if capture else None
        env = {"COLI_PREFIX_LOG": "1", "COLI_SPEC_GATE": "0",
               "COLI_STRATA_INPUT": "next-embedding", "COLI_LOOKUP": "0",
               "COLI_STRATA_HEADS": "", "COLI_STRATA_TRACE": "", "COLI_LOOKUP_FORCE": "",
               "Q36_MTP": "", "Q36_MTP_DRAFTS": str(depth or getattr(args, "mtp_drafts", 3)),
               "Q36_MTP_FORCE": "", "Q36_MTP_TEST_ORACLE": "",
               "COLI_CUDA": "0", "COLI_VULKAN": "0"}
        require(not (bank and mtp), "MTP and CSNH fixture adapters must be separate")
        if mtp:
            env["Q36_MTP"] = linux_path(args.mtp_dir) if args.wsl else str(args.mtp_dir)
            env["Q36_MTP_FORCE"] = force
            if oracle:
                env["Q36_MTP_TEST_ORACLE"] = linux_path(oracle) if args.wsl else str(oracle)
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
        frames, bodies, ids, cancelled, prompt_tokens = [], [], [], False, None
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
            if fields[0] == b"ACCEPT":
                require(len(fields) == 3 and fields[2].isdigit(), "missing prompt-token evidence")
                prompt_tokens = int(fields[2])
            elif fields[0] == b"DATA":
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
                return {"id": rid, "ids": ids, "frames": frames, "body": b"".join(bodies), "cancelled": True,
                        "token_ids_verified": temperature == 0}
            elif fields[0] == b"DONE":
                require(not cancelled, "cancellation arrived after the request finished")
                require(fields[2] == b"STAT" and int(fields[3]) == len(ids), "bad DONE token accounting")
                return {"id": rid, "ids": ids, "frames": frames, "body": b"".join(bodies), "cancelled": False,
                        "token_ids_verified": temperature == 0, "prompt_tokens": prompt_tokens,
                        "limited": bool(int(fields[8])) if len(fields) >= 9 else None}

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
    pattern = MTP_STATS if getattr(driver, "mtp_active", False) else STATS
    match = next((m for line in driver.log.splitlines() if (m := pattern.fullmatch(line)) and m[1] == rid), None)
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
    require(args.bank_kind in ("ordinary", "logit-residual"), "unsupported fixture bank kind")
    require(1 <= len(tokens) <= 8 and all(type(token) is int and 0 <= token < vocab for token in tokens),
            "fixture needs 1..8 valid vocabulary token IDs")
    residual = args.bank_kind == "logit-residual"
    identity = {"family": "qwen36", "checkpoint_revision": args.revision,
                "tokenizer_sha256": native["tokenizer_sha256"], "native_fingerprint": native["fingerprint"],
                "hidden_tap": exporter.AUGMENTED_TAP, "quantization": PRECISION,
                "hidden_size": width, "vocab_size": vocab}
    pack = {"schema": RESIDUAL_SCHEMA if residual else exporter.heads.PACK_SCHEMA, "identity": identity, "heads": [],
            "training": {"synthetic_correctness_fixture_only": True, "quality_validated": False}}
    if residual:
        pack["kind"] = "logit-residual"
        pack["training"]["bias_policy"] = "finite +/-1e6 delta dominates ordinary target logits; correctness only"
    for index, token in enumerate(tokens):
        bias = [-RESIDUAL_FIXTURE_BIAS if residual else 0.0] * vocab
        bias[token] = RESIDUAL_FIXTURE_BIAS if residual else 1.0
        pack["heads"].append({"future_offset": index + 2, "rank": 1,
                              "down": [0.0] * width, "up": [0.0] * vocab, "bias": bias})
    path = args.work / (name + ".snh")
    blob = exporter.export_pack(pack, args.model, PRECISION, bits=4, kind=args.bank_kind)
    require(len(blob) >= 96, "exporter returned a truncated fixture bank")
    header = struct.unpack_from("<4s7I", blob)
    require(header[0] == b"CSNH" and (header[1], header[7]) == ((2, 1) if residual else (1, 0)),
            "exported bank kind does not match the requested fixture mode")
    with path.open("xb") as stream:
        stream.write(blob)
    return path


def paired_case(args, name, bank=None, budget=10, continuation=False, pin=False, sample=False, prompt=PROMPT,
                *, mtp=False, force="", oracle=None, depth=None):
    gold = feature = None
    try:
        gold = Driver(args, name + "-plain")
        feature = Driver(args, name + "-heads", bank, mtp=mtp, force=force, oracle=oracle, depth=depth)
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
        require(("[qwen36 MTP] pretrained sequential drafter ready" if mtp else "[strata] loaded") in feature.log,
                "sample fallback never loaded the test adapter")
        require(not any((MTP_STATS if mtp else STATS).fullmatch(line) for line in feature.log.splitlines()),
                "sampled request executed unqualified head path")
        return {"name": name, "tokens": len(b["ids"]), "frontier_rows": rows, "sample_fallback": True,
                "pretrained_mtp": mtp, "mtp_depth": (depth or getattr(args, "mtp_drafts", None)) if mtp else None,
                "sample_token_ids_verified": False,
                "sample_evidence_scope": "exact chosen logprob, top-token DATA, emitted bytes and hidden frontier; sampled chosen ID is absent from protocol"}
    values = stats(feature)
    require(values["bank_calls"] > 0 and values["failures"] == 0, "bank was inactive or failed")
    if name == "first-reject":
        require(values["accepted"] == 0, "first-rejection fixture unexpectedly accepted a draft")
    if name == "middle-reject":
        require(values["accepted"] >= 1 and values["drafts"] > values["accepted"], "middle rejection was not reached")
    if name == "cycle-rejections":
        require(values["verifies"] >= 2 and values["accepted"] > 0 and values["drafts"] > values["accepted"],
                "rotating acceptance/rejection paths were not reached")
    if name == "full-accept-next-proposal":
        require(values["accepted"] >= 2 and values["bank_calls"] >= 2, "full acceptance/new proposal was not reached")
    if name == "budget-clipped-verify":
        drafts, forwards = (2, 1) if mtp else (1, 2)
        require(values["drafts"] == drafts and values["accepted"] == drafts and
                values["forwards"] == forwards and values["tokens"] == 3,
                "budget did not clip the proposed chain to consumable verifier rows")
        require(not ended_ahead(feature), "fixed output budget speculated into the final unfed token")
    if name == "eos-mid-verify":
        require(2 <= len(b["ids"]) < budget and ended_ahead(feature),
                "EOS did not end generation with pending verifier rows")
    if continuation:
        require("[PREFIX] reusing" in feature.log, "continuation did not exercise live prefix reuse")
    if pin:
        require("[PIN] scatto" in feature.log and "[PIN] restoring" in feature.log, "pin restore path was not observed")
    return {"name": name, "tokens": len(b["ids"]), "frontier_rows": rows, "stats": values,
            "pretrained_mtp": mtp, "mtp_depth": depth or getattr(args, "mtp_drafts", None) if mtp else None,
            "forced_proposals_correctness_only": bool(force)}


def cancellation_case(args, bank=None, *, mtp=False, oracle=None):
    gold = feature = None
    try:
        gold = Driver(args, "cancel-plain")
        feature = Driver(args, "cancel-heads", bank, mtp=mtp, force="accept" if mtp else "", oracle=oracle, depth=3 if mtp else None)
        a = gold.ask(PROMPT, 12)
        threshold = 2 if mtp else 3
        b = feature.ask(PROMPT, 12, cancel_after=threshold)
        require(b["cancelled"] and threshold <= len(b["ids"]) < 12, "cancellation was not observed during generation")
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
            "post_cancel_frontier_rows": len(feature_next), "stats": values,
            "pretrained_mtp": mtp, "forced_proposals_correctness_only": mtp}


def precision_refusals(args, bank=None, *, mtp=False):
    results = []
    for key, value in (("COLI_DENSE_IDOT", "0"), ("COLI_DENSE_BITS", "4"), ("COLI_DENSE_I8", "0")):
        env = {"SNAP": linux_path(args.model) if args.wsl else str(args.model), "SERVE": "1",
               "OMP_NUM_THREADS": str(args.threads), "COLI_NO_OMP_TUNE": "1", "COLI_LOOKUP": "0",
               "COLI_STRATA_INPUT": "next-embedding", "COLI_DENSE_IDOT": "1", "COLI_DENSE_BITS": "8",
               "COLI_DENSE_I8": "1", "COLI_STRATA_HEADS": (linux_path(bank) if args.wsl else str(bank)) if bank else "",
               "COLI_STRATA_TRACE": "", "Q36_MTP": (linux_path(args.mtp_dir) if args.wsl else str(args.mtp_dir)) if mtp else "",
               "Q36_MTP_DRAFTS": "3", "Q36_MTP_FORCE": "", "Q36_MTP_TEST_ORACLE": "",
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
        require((b"requires qualified dense CPU INT8/IDOT" if mtp else b"initial native qualification requires") in result.stderr,
                "wrong precision failed for an unrelated reason")
        results.append({"name": "precision-refusal", "setting": key + "=" + value, "exit_code": result.returncode})
    return results


def parse_args(argv=None):
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
    parser.add_argument("--bank-kind", choices=("ordinary", "logit-residual"), default="ordinary",
                        help="synthetic fixture bank format; residual uses finite +/-1e6 delta biases")
    parser.add_argument("--mtp-dir", type=Path, help="authentic pretrained MTP sidecar; selects the separate MTP harness")
    parser.add_argument("--mtp-drafts", type=int, choices=(1, 2, 3), default=3,
                        help="default MTP depth; forced rollback fixtures use 3, actual-depth cases use their named depth")
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 64 and 1 <= args.request_timeout <= 600, "invalid execution bounds")
    require(re.fullmatch(r"[0-9a-f]{40}", args.revision), "revision must be 40 lowercase hex digits")
    require(not os.path.lexists(args.work) and not os.path.lexists(args.output), "work directory and report must be new")
    args.engine, args.model, args.work, args.output = (p.resolve() for p in (args.engine, args.model, args.work, args.output))
    require(args.engine.is_file() and args.model.is_dir(), "native engine/model is absent")
    require(not os.path.lexists(args.work) and not os.path.lexists(args.output), "work directory and report must be new")
    require(not args.work.is_relative_to(args.model) and not args.output.is_relative_to(args.model),
            "work and report must be outside model assets")
    available = CASE_NAMES
    if args.mtp_dir:
        args.mtp_dir = args.mtp_dir.resolve()
        require(args.mtp_dir.is_dir() and all((args.mtp_dir / name).is_file() for name in ("model-mtp.safetensors", "qwen36_mtp.json")), "MTP sidecar is absent")
        require(args.bank_kind == "ordinary", "MTP and residual-bank harness modes are separate")
        require(not args.work.is_relative_to(args.mtp_dir) and not args.output.is_relative_to(args.mtp_dir), "work and report must be outside MTP assets")
        available = MTP_CASE_NAMES
    selected = args.cases.split(",") if args.cases is not None else list(available)
    require(selected and len(selected) == len(set(selected)) and set(selected) <= set(available),
            "unknown, duplicate or empty case selection")
    args.selected_cases = set(selected)
    return args


def mtp_oracle(args, name, result, extra_token=None):
    require(result.get("token_ids_verified") and type(result.get("prompt_tokens")) is int and result["prompt_tokens"] > 0,
            "forced MTP fixture requires a genuine greedy oracle and ACCEPT prompt count")
    tokens = list(result["ids"])
    if extra_token is not None: tokens.append(extra_token)
    require(1 <= len(tokens) <= 512 and all(type(token) is int and token >= 0 for token in tokens), "invalid oracle token IDs")
    payload = json.dumps({"prompt_tokens": result["prompt_tokens"], "tokens": tokens}).encode("ascii")
    require(len(payload) <= 65536, "oracle exceeds native byte limit")
    path = args.work / (name + ".json")
    with path.open("xb") as stream: stream.write(payload)
    return path


def mtp_cases(args, native, truth):
    selected, records = args.selected_cases, []
    oracle = mtp_oracle(args, "synthetic-proposal-oracle", truth)
    for name, mode, options in (
        ("first-reject", "row1", {"continuation": True}),
        ("middle-reject", "row2", {"continuation": True}),
        ("cycle-rejections", "cycle", {"continuation": True}),
        ("full-accept-next-proposal", "accept", {"continuation": True}),
        ("budget-clipped-verify", "accept", {"budget": 4, "continuation": True}),
        ("prefix-pin-restore", "accept", {"continuation": True, "pin": True}),
        ("sample-fallback", "", {"sample": True}),
        ("actual-depth1", "", {"depth": 1, "continuation": True}),
        ("actual-depth2", "", {"depth": 2, "continuation": True}),
        ("actual-depth3", "", {"depth": 3, "continuation": True}),
    ):
        if name not in selected: continue
        print(json.dumps({"name": name, "status": "running"}), flush=True)
        record = paired_case(args, name, mtp=True, force=mode, oracle=oracle if mode else None,
                             **({"depth": 3 if mode else args.mtp_drafts, **options}))
        records.append(record); print(json.dumps(record), flush=True)
    if "cancel-mid-verify" in selected:
        record = cancellation_case(args, mtp=True, oracle=oracle)
        records.append(record); print(json.dumps(record), flush=True)
    if "precision-refusal" in selected:
        records.extend(precision_refusals(args, mtp=True))
    if "eos-mid-verify" in selected:
        gold = Driver(args, "eos-oracle", capture=False)
        try: result = gold.ask(EOS_PROMPT, 20)
        finally: gold.close()
        require(result["limited"] is False and 2 <= len(result["ids"]) < 20, "EOS oracle did not reach a natural early stop")
        config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        eos = config.get("text_config", config).get("eos_token_id")
        require(type(eos) is int and 0 <= eos < native["vocab_size"], "model lacks scalar EOS")
        eos_oracle = mtp_oracle(args, "synthetic-eos-oracle", result, eos)
        record = paired_case(args, "eos-mid-verify", mtp=True, force="accept", oracle=eos_oracle,
                             depth=3, prompt=EOS_PROMPT, budget=20, continuation=True)
        records.append(record); print(json.dumps(record), flush=True)
    return records


def main(argv=None):
    args = parse_args(argv)
    args.work.mkdir(parents=True)
    started = time.monotonic()
    oracle = Driver(args, "oracle", capture=False)
    try:
        truth_result = oracle.ask(PROMPT, 12)
        truth = truth_result["ids"]
    finally:
        oracle.close()
    require(len(truth) >= 10, "oracle stopped too early for frontier gates")
    native = exporter.fingerprint_info(args.model, PRECISION)
    if args.mtp_dir:
        records = mtp_cases(args, native, truth_result)
        report = {"schema": "colibri.native-pretrained-mtp-frontier-correctness/v1", "verdict": "approved",
                  "performance_evidence": False, "authentic_pretrained_cell_requested": True,
                  "authentic_pretrained_cell_execution_observed": any(record.get("stats", {}).get("bank_calls", 0) > 0 for record in records),
                  "forced_proposals_correctness_only": any(record.get("forced_proposals_correctness_only", False) for record in records),
                  "selected_cases": sorted(args.selected_cases),
                  "forced_oracle_scope": "initial prompt length only; continuation uses actual cell proposals",
                  "engine_sha256": sha(args.engine), "native_fingerprint": native["fingerprint"],
                  "mtp_identity": {name: sha(args.mtp_dir / name) for name in ("model-mtp.safetensors", "qwen36_mtp.json")},
                  "runtime_precision": PRECISION, "oracle_ids": truth, "records": records,
                  "elapsed_seconds": time.monotonic() - started}
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2); stream.write("\n")
        print("pretrained MTP frontier: real cell, exact output/hidden parity and selected rollback gates passed", flush=True)
        return 0
    wrong = next(token for token in range(native["vocab_size"]) if token not in truth)
    reject = constant_bank(args, native, "constant-reject", [wrong])
    middle = constant_bank(args, native, "constant-middle", [truth[2], wrong])
    full = constant_bank(args, native, "constant-full", truth[2:4])
    deep = constant_bank(args, native, "constant-deep", truth[2:7])
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    eos_id = config.get("text_config", config).get("eos_token_id")
    require(type(eos_id) is int and 0 <= eos_id < native["vocab_size"], "model lacks a scalar EOS token")
    eos = constant_bank(args, native, "constant-eos", [eos_id])
    selected = args.selected_cases
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
              "bank_kind": args.bank_kind, "bank_format": "CSNH v2/reserved=1" if args.bank_kind == "logit-residual" else "CSNH v1/reserved=0",
              "fixture_bias": {"selected": RESIDUAL_FIXTURE_BIAS, "others": -RESIDUAL_FIXTURE_BIAS} if args.bank_kind == "logit-residual" else {"selected": 1.0, "others": 0.0},
              "residual_fixture_scope": ("dominant synthetic deltas test transaction/frontier correctness; exact raw-anchor values require the separate C anchor gate"
                                         if args.bank_kind == "logit-residual" else None),
              "selected_cases": sorted(selected),
              "elapsed_seconds": time.monotonic() - started}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("native head frontier: active-bank rollback, hidden traces, protocol and numeric-mode gates passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure native CPU optimizations, then isolate independent parallel heads.

The stock/candidate study keeps heads OFF and prompt lookup ON. Optional
same-binary studies disable one new knob at a time; the optional head study
compares the SAME optimized engine and settings with its bank OFF/ON.
Correctness requests carry token/logprob evidence. Timed requests omit that
channel. Processes execute one arm at a time in alternating order.

This tool downloads nothing and does not infer hardware qualification from
environment settings. A fresh process is not evidence of a cold OS page cache.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

try:
    from .native_decode_bench import NativeEngine, bounded_int, bounded_number, file_sha, linux_path
except ImportError:
    from native_decode_bench import NativeEngine, bounded_int, bounded_number, file_sha, linux_path

MAX_PROMPTS_BYTES = 256 * 1024
MAX_PROMPT_BYTES = 65536
STUDY_IDS = ("stock", "rows", "group", "wide", "heads")


def parse_profile(line):
    """Qwen's untagged PROF frame belongs to its current single-slot request."""
    fields = line.split()
    if len(fields) != 10 or fields[0] != b"PROF":
        raise ValueError("invalid native PROF frame")
    seconds = [bounded_number(float(fields[i]), 0, 1e9, "profile seconds") for i in (1, 4, 5, 6, 7, 8)]
    return {"decode_seconds": seconds[0], "prompt_tokens": bounded_int(int(fields[2]), 0, 2**31-1, "prompt tokens"),
            "completion_tokens": bounded_int(int(fields[3]), 0, 512, "completion tokens"),
            "disk_seconds": seconds[1], "wait_seconds": seconds[2], "moe_expert_seconds": seconds[3],
            "attention_seconds": seconds[4], "output_head_seconds": seconds[5],
            "phase_scope": {"moe_expert_seconds": "native tm_sum(2), less disk time, clipped at zero; excludes other dense/GDN matrix work",
                            "attention_seconds": "native tm_sum(0)+tm_sum(1)",
                            "output_head_seconds": "native tm_sum(5): vocabulary projection; independent draft-bank cost is not isolated",
                            "wait_seconds": "native PROF placeholder currently emitted as zero"},
            "forwards": bounded_int(int(fields[9]), 0, 2**31-1, "forwards")}


class MetricEngine(NativeEngine):
    """Observe existing protocol fields without changing the shared driver."""
    def __init__(self, *args, **kwargs):
        self._observations = None
        overrides = kwargs.get("env") or (args[3] if len(args) > 3 else {}) or {}
        self.phase_timers_enabled = overrides.get("COLI_TIMERS") == "1"
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        self.startup_ready_seconds = time.perf_counter() - started

    def _next(self, deadline):
        line, body = super()._next(deadline)
        if self._observations is not None:
            if line.startswith(b"PROF "):
                self._observations["profiles"].append(parse_profile(line))
            elif line.startswith(b"ACCEPT "):
                fields = line.split()
                if len(fields) >= 3:
                    self._observations["prompt_tokens"] = bounded_int(int(fields[2]), 0, 2**31-1, "prompt tokens")
        return line, body

    def turn(self, *args, **kwargs):
        self._observations = {"profiles": [], "prompt_tokens": None}
        try:
            result = super().turn(*args, **kwargs)
            profiles = self._observations["profiles"]
            prompt_tokens = self._observations["prompt_tokens"]
            if len(profiles) > 1:
                raise ValueError("multiple PROF frames in a single-slot request")
            profile = profiles[0] if profiles else None
            if profile and (profile["completion_tokens"] != result["tokens"] or
                            (prompt_tokens is not None and profile["prompt_tokens"] != prompt_tokens)):
                raise ValueError("PROF/ACCEPT/DONE counts differ")
            result["protocol_profile"] = profile
            result["phase_timers_enabled"] = self.phase_timers_enabled
            result["phase_seconds_available"] = bool(profile) and self.phase_timers_enabled
            result["phase_seconds_unavailable_reason"] = (None if result["phase_seconds_available"] else
                "native phase timers disabled" if not self.phase_timers_enabled else "no PROF frame emitted")
            result["prompt_tokens"] = prompt_tokens
            result["prefill_seconds"] = None
            result["prefill_tokens_per_second"] = None
            result["prefill_unavailable_reason"] = "protocol has no isolated prefill duration"
            first, accepted = result.get("ttft_seconds"), result.get("accept_seconds")
            result["observed_prompt_to_first_data_seconds"] = (first - accepted if first is not None and accepted is not None else None)
            result["observed_prompt_interval_scope"] = "client observation including prefill, first-token work and pipe delivery"
            return result
        finally:
            self._observations = None


def common_env(args):
    return {"COLI_LOOKUP": "1", "COLI_LOOKUP_FORCE": "", "COLI_LOOKUP_DRAFTS": "", "COLI_SPEC_GATE": "1",
            "COLI_CUDA": "0", "COLI_VULKAN": "0", "Q36_DN_GPU": "0", "CACHE_ROUTE": "0",
            "COLI_DENSE_BITS": "8", "COLI_DENSE_I8": "1", "COLI_DENSE_IDOT": "1",
            "COLI_IDOT_WIDE": "0",
            "Q36_MAXT": "2048", "COLI_KV_PREFIX": "1", "QWEN_DENSE_BATCH": "1", "QWEN36_QPACK": "",
            "COLI_STRATA_HEADS": "", "COLI_STRATA_TRACE": "", "COLI_STRATA_INPUT": "",
            # Keep the prior fork's head batching and expert chunks in every
            # same-binary ablation; only its selected new knob varies.
            "QWEN_HEAD_BATCH": "1", "QWEN_EXPERT_BATCH": "1",
            "COLI_TIMERS": "1" if args.phase_timers else "0"}


def arm(args, name, engine, rows, projection_group, heads=False, source_kind=None, wide=None):
    env = common_env(args)
    wide = args.wide if wide is None else wide
    env.update(COLI_IDOT_ROWS=str(rows), QWEN_PROJECTION_GROUP=str(projection_group), COLI_IDOT_WIDE=str(wide))
    if heads:
        env["COLI_STRATA_HEADS"] = linux_path(args.heads) if args.wsl else str(args.heads)
        env["COLI_STRATA_INPUT"] = args.head_input
    return {"name": name, "engine": str(engine), "source_kind": source_kind or args.optimized_source_kind,
            "source_kind_verification": "caller-supplied binary provenance; file hashes recorded",
            "rows": rows, "projection_group": projection_group, "wide": wide,
            "wide_setting_scope": "requested kernel setting; unsupported ISA uses the ordinary native kernel",
            "parallel_heads": heads, "env": env}


def study_plan(args):
    optimized = arm(args, "optimized", args.optimized, args.rows, args.projection_group)
    studies = {
        "stock": {"id": "stock", "name": "stock-vs-optimized-heads-off", "purpose": "all candidate CPU changes against unchanged stock binary",
                  "control": arm(args, "stock", args.baseline, 0, 0, source_kind=args.baseline_source_kind, wide=0),
                  "candidate": optimized},
        "rows": {"id": "rows", "name": "row-tiling-ablation", "purpose": "same optimized binary; only COLI_IDOT_ROWS changes",
             "control": arm(args, "rows-off", args.optimized, 0, args.projection_group),
             "candidate": arm(args, "rows-on", args.optimized, 1, args.projection_group)},
        "group": {"id": "group", "name": "projection-group-ablation", "purpose": "same optimized binary; only QWEN_PROJECTION_GROUP changes",
             "control": arm(args, "group-off", args.optimized, args.rows, 0),
             "candidate": arm(args, "group-on", args.optimized, args.rows, 1)},
        "wide": {"id": "wide", "name": "wide-integer-dot-ablation", "purpose": "same optimized binary; only COLI_IDOT_WIDE changes",
                 "control": arm(args, "wide-off", args.optimized, args.rows, args.projection_group, wide=0),
                 "candidate": arm(args, "wide-on", args.optimized, args.rows, args.projection_group, wide=1)},
    }
    if args.heads:
        studies["heads"] = {"id": "heads", "name": "parallel-head-contribution", "purpose": "same optimized binary and CPU knobs; only independent head bank changes",
                            "control": optimized,
                            "candidate": arm(args, "optimized-with-heads", args.optimized, args.rows, args.projection_group, True)}
    selected = args.studies
    if selected is None:
        selected = ["stock"]
        if args.ablate_knobs: selected.extend(("rows", "group"))
        if args.ablate_wide: selected.append("wide")
        if args.heads: selected.append("heads")
    return [studies[key] for key in selected]


def model_identity(root):
    paths = sorted((p for p in root.iterdir() if p.is_file() and
                    (p.name in ("config.json", "tokenizer.json", "qwen36_meta.json", "model.safetensors.index.json")
                     or p.suffix == ".safetensors")),
                   key=lambda p: p.name.encode("utf-8"))
    labels = {p.name for p in paths}
    if not {"config.json", "tokenizer.json"} <= labels or not any(p.suffix == ".safetensors" for p in paths):
        raise ValueError("native model needs config.json, tokenizer.json and safetensors weights")
    if len(paths) > 4096:
        raise ValueError("model manifest has too many files")
    files = []
    for path in paths:
        before = path.stat()
        digest = file_sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("model asset changed during identity hashing")
        files.append({"name": path.name, "bytes": before.st_size, "sha256": digest})
    return {"path": str(root), "files": files, "total_bytes": sum(f["bytes"] for f in files),
            "manifest_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "weights_hashed": True, "hashing_precedes_measurement_and_warms_page_cache": True}


def normalize_frames(result):
    return [line.split(" ", 2)[2] for line in result["numeric_frames"]]


def equal_output(a, b):
    return a["tokens"] == b["tokens"] and a["output_sha256"] == b["output_sha256"]


def check_numeric(a, b):
    return (a.get("token_ids_verified") is True and b.get("token_ids_verified") is True and
            len(a["token_ids"]) == a["tokens"] and len(b["token_ids"]) == b["tokens"] and
            len(a["numeric_frames"]) == a["tokens"] and len(b["numeric_frames"]) == b["tokens"] and
            equal_output(a, b) and a["token_ids"] == b["token_ids"] and normalize_frames(a) == normalize_frames(b))


def timed_valid(result):
    bounded_int(result["tokens"], 1, 64, "timed completed tokens")
    bounded_number(result["engine_decode_tokens_per_second"], 0, 1e9, "decode rate", True)
    bounded_number(result["wall_seconds"], 0, 3600, "request wall", True)
    if result.get("ttft_seconds") is not None:
        bounded_number(result["ttft_seconds"], 0, 3600, "TTFT")


def summarize(records):
    def cells(rows):
        ratios = [r["candidate"]["engine_decode_tokens_per_second"] / r["control"]["engine_decode_tokens_per_second"] for r in rows]
        return {"pairs": len(rows), "median_paired_decode_ratio": statistics.median(ratios),
                "paired_decode_ratio_min": min(ratios), "paired_decode_ratio_max": max(ratios),
                "control_median_decode_tokens_per_second": statistics.median(r["control"]["engine_decode_tokens_per_second"] for r in rows),
                "candidate_median_decode_tokens_per_second": statistics.median(r["candidate"]["engine_decode_tokens_per_second"] for r in rows),
                "median_paired_wall_speed_ratio": statistics.median(r["control"]["wall_seconds"] / r["candidate"]["wall_seconds"] for r in rows)}
    overall = cells(records)
    overall["per_prompt"] = [{"prompt": i, **cells([r for r in records if r["prompt"] == i])} for i in range(3)]
    return overall


def run_study(args, prompts, definition, factory, report, checkpoint=lambda: None):
    result = {**copy.deepcopy(definition), "status": "running", "correctness": [],
              "fresh_process_requests": [], "warmups": [], "records": []}
    report["studies"].append(result)
    engines = {}
    try:
        for side in ("control", "candidate"):
            settings = result[side]
            engine = factory(settings["engine"], args.model, args.threads, settings["env"], args.wsl,
                             startup_timeout=args.startup_timeout)
            engines[side] = engine
            settings["observed_launch_settings"] = getattr(engine, "launch_settings", {})
            settings["startup_ready_seconds"] = getattr(engine, "startup_ready_seconds", None)
            checkpoint()
        for index, prompt in enumerate(prompts):
            # First request per process is retained separately. Both processes
            # share an uncontrolled, preflight-hashed OS page cache.
            if index == 0:
                pair = {side: engine.turn(prompt, 64, False, timeout=args.request_timeout) for side, engine in engines.items()}
                result["fresh_process_requests"].append({"prompt": index, **pair, "page_cache_cold_verified": False})
                checkpoint()
                for value in pair.values(): timed_valid(value)
                if not equal_output(pair["control"], pair["candidate"]):
                    raise RuntimeError("fresh-process output differs")
            pair = {side: engine.turn(prompt, 64, True, timeout=args.request_timeout) for side, engine in engines.items()}
            equal = check_numeric(pair["control"], pair["candidate"])
            result["correctness"].append({"prompt": index, "equal": equal, **pair})
            checkpoint()
            if not equal:
                raise RuntimeError(f"token ID/logprob/output mismatch for prompt {index}")
            expected = pair["control"]
            warm = {side: engine.turn(prompt, 64, False, timeout=args.request_timeout) for side, engine in engines.items()}
            result["warmups"].append({"prompt": index, **warm})
            checkpoint()
            if not all(equal_output(value, expected) for value in warm.values()):
                raise RuntimeError("warmup output changed from the numeric oracle")
            for repeat in range(args.repeats):
                order = ("control", "candidate") if (index + repeat) % 2 == 0 else ("candidate", "control")
                pair = {side: engines[side].turn(prompt, 64, False, timeout=args.request_timeout) for side in order}
                result["records"].append({"prompt": index, "repeat": repeat, "order": list(order), **pair})
                checkpoint()
                for value in pair.values(): timed_valid(value)
                if not all(equal_output(value, expected) for value in pair.values()):
                    raise RuntimeError("timed output/count changed from the numeric oracle")
                print(json.dumps({"study": definition["name"], "prompt": index, "repeat": repeat,
                                  "control_tps": pair["control"]["engine_decode_tokens_per_second"],
                                  "candidate_tps": pair["candidate"]["engine_decode_tokens_per_second"]}), flush=True)
        result["summary"] = summarize(result["records"])
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        cleanup_error = None
        for side, engine in engines.items():
            try:
                engine.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
                result["status"] = "failed"
                result.setdefault("cleanup_errors", []).append({"arm": side, "type": type(error).__name__, "message": str(error)})
            result[side + "_stderr"] = list(engine.stderr)
            result[side + "_stderr_truncated_bytes"] = getattr(engine, "stderr_truncated_bytes", 0)
            result[side + "_ready"] = getattr(engine, "ready_line", None)
        checkpoint()
        if cleanup_error is not None and result.get("error") is None:
            raise cleanup_error


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_value": repr(value)}
    if isinstance(value, dict): return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe_json(v) for v in value]
    return value


def validate_args(args):
    bounded_int(args.threads, 1, 64, "threads")
    bounded_int(args.repeats, 5, 100, "repeats")
    if args.tokens != 64: raise ValueError("qualification requires a fixed output cap of 64 tokens")
    bounded_number(args.request_timeout, 0, 600, "request timeout", True)
    bounded_number(args.startup_timeout, 0, 600, "startup timeout", True)
    if args.studies is not None:
        selected = args.studies.split(",")
        if not selected or len(selected) > len(STUDY_IDS) or len(set(selected)) != len(selected) or any(key not in STUDY_IDS for key in selected):
            raise ValueError("studies must be distinct comma-separated IDs: stock,rows,group,wide,heads")
        if "heads" in selected and not args.heads:
            raise ValueError("the heads study requires --heads")
        args.studies = selected
    for label in ("baseline_source_kind", "optimized_source_kind"):
        value = getattr(args, label)
        if not value or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise ValueError("source kind must be a bounded description")
    for label in ("baseline", "optimized", "prompts"):
        path = getattr(args, label).resolve()
        if not path.is_file(): raise ValueError(label + " must be an existing file")
        setattr(args, label, path)
    args.model = args.model.resolve()
    if not args.model.is_dir(): raise ValueError("model directory is absent")
    if args.heads:
        args.heads = args.heads.resolve()
        if not args.heads.is_file(): raise ValueError("head bank is absent")
    if os.path.lexists(args.output): raise FileExistsError("report must be a new path")
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.model): raise ValueError("report must be outside model assets")
    if args.prompts.stat().st_size > MAX_PROMPTS_BYTES: raise ValueError("prompt file exceeds 256 KiB")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if (not isinstance(prompts, list) or len(prompts) != 3 or
            not all(isinstance(p, str) and p and len(p.encode("utf-8")) <= MAX_PROMPT_BYTES for p in prompts) or
            len(set(prompts)) != 3):
        raise ValueError("provide exactly three distinct bounded held-out prompt strings")
    return prompts


def main(argv=None, *, engine_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--baseline-source-kind", default="unchanged upstream Colibri native CPU binary")
    parser.add_argument("--optimized-source-kind", default="Colibri fork native CPU: row tiling, grouped projections and optional wide integer-dot")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--head-input", choices=("hidden", "next-embedding"), default="next-embedding")
    parser.add_argument("--rows", type=int, choices=(0, 1), default=1)
    parser.add_argument("--wide", type=int, choices=(0, 1), default=0)
    parser.add_argument("--projection-group", "--shared-act", dest="projection_group", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ablate-knobs", action="store_true")
    parser.add_argument("--ablate-wide", action="store_true")
    parser.add_argument("--studies", help="select only these comma-separated studies in this order: stock,rows,group,wide,heads; selection overrides ablation flags")
    parser.add_argument("--phase-timers", action="store_true", help="enable identical native phase timers in both arms; adds instrumentation")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--wsl")
    args = parser.parse_args(argv)
    prompts = validate_args(args)
    definitions = study_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "colibri.native-cpu-optimization-benchmark/v1", "status": "running", "studies": [],
              "settings": {"threads": args.threads, "repeats": args.repeats, "fixed_output_cap": 64,
                           "lookup": True, "temperature": 0, "top_p": 1, "wsl": args.wsl,
                           "phase_timers": args.phase_timers, "rows": args.rows, "projection_group": args.projection_group,
                           "wide": args.wide, "studies": [definition["id"] for definition in definitions]},
              "held_out_status": "caller-supplied three distinct prompts; training disjointness is not inferred",
              "requested_compute": "CPU dense INT8/integer-dot", "hardware_execution_qualified": False,
              "cold_page_cache_verified": False, "prefill_metric_available": False,
              "prefill_metric_reason": "native protocol reports prompt count but no isolated prefill duration"}
    report["invocation"] = {"executable": sys.executable, "script": str(Path(__file__).resolve()),
                            "arguments": list(argv) if argv is not None else sys.argv[1:]}
    failure = None
    # Claim and keep one exclusive descriptor: failures never overwrite a
    # pre-existing artifact, and partial evidence is retained at this path.
    with args.output.open("x", encoding="utf-8") as output:
        def checkpoint():
            output.seek(0)
            json.dump(safe_json(report), output, indent=2, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()
        try:
            report["model_identity"] = model_identity(args.model)
            report["binaries"] = {"baseline": {"path": str(args.baseline), "sha256": file_sha(args.baseline)},
                                  "optimized": {"path": str(args.optimized), "sha256": file_sha(args.optimized)}}
            report["prompt_sha256"] = file_sha(args.prompts)
            report["head_identity"] = {"path": str(args.heads), "sha256": file_sha(args.heads), "input": args.head_input} if args.heads else None
            checkpoint()
            for definition in definitions:
                run_study(args, prompts, definition, engine_factory or MetricEngine, report, checkpoint)
            for label in ("baseline", "optimized"):
                if file_sha(getattr(args, label)) != report["binaries"][label]["sha256"]:
                    raise RuntimeError("engine binary changed during measurement")
            if args.heads and file_sha(args.heads) != report["head_identity"]["sha256"]:
                raise RuntimeError("head bank changed during measurement")
            report["model_identity_after"] = model_identity(args.model)
            if report["model_identity_after"]["manifest_sha256"] != report["model_identity"]["manifest_sha256"]:
                raise RuntimeError("model assets changed during measurement")
            report["model_identity_verified_before_and_after"] = True
            report["status"] = "complete"
        except BaseException as error:
            failure = error
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            checkpoint()
    if failure is not None: raise failure
    return report


if __name__ == "__main__":
    main()

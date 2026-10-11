#!/usr/bin/env python3
"""Paired OLMoE CPU expert-prefetch qualification with a fixed cache budget.

Uses existing Colibri PILOT=0/1, not a newly ported GPU backend. DONE's rate
includes prefill, generation and request-owned prefetch drain. Hashing warms
the OS page cache; this is not a cold SSD/NVMe measurement. Downloads nothing.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import time

try:
    from .native_decode_bench import NativeEngine, bounded_int, bounded_number, file_sha
    from .native_optimization_bench import check_numeric, equal_output, model_identity, parse_profile, safe_json
except ImportError:
    from native_decode_bench import NativeEngine, bounded_int, bounded_number, file_sha
    from native_optimization_bench import check_numeric, equal_output, model_identity, parse_profile, safe_json


class MoeMetricEngine(NativeEngine):
    """OLMoE emits its untagged PROF after DONE; consume it in the same turn."""
    def __init__(self, *args, **kwargs):
        self._pending_done = None
        self._profile = None
        self._done_stats = None
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        self.startup_ready_seconds = time.perf_counter() - started

    def _next(self, deadline):
        if self._pending_done is not None:
            done = self._pending_done
            self._pending_done = None
            return done
        line, body = super()._next(deadline)
        if line.startswith(b"PROF "):
            raise ValueError("OLMoE PROF arrived without its preceding DONE")
        if line.startswith(b"DONE "):
            fields = line.split()
            if len(fields) != 9 or fields[2] != b"STAT":
                raise ValueError("invalid OLMoE DONE frame")
            self._done_stats = {
                "cache_hit_percent": bounded_number(float(fields[5]), 0, 100, "cache hit"),
                "rss_gb": bounded_number(float(fields[6]), 0, 1e6, "RSS"),
                "prompt_tokens": bounded_int(int(fields[7]), 0, 2048, "prompt tokens"),
                "length_limited": bounded_int(int(fields[8]), 0, 1, "length limited")}
            profile_line, _ = super()._next(deadline)
            profile = parse_profile(profile_line)
            if profile["completion_tokens"] != int(fields[3]) or profile["prompt_tokens"] != self._done_stats["prompt_tokens"]:
                raise ValueError("OLMoE PROF/DONE counts differ")
            profile["request_seconds"] = profile.pop("decode_seconds")
            profile["phase_scope"] = {
                "request_seconds": "prefill, generation, and unused prefetch drain",
                "disk_seconds": "aggregate expert-read wall durations including worker; overlaps computation",
                "moe_expert_seconds": "MoE block wall less aggregate read wall, clipped at zero",
                "attention_seconds": "native attention wall accumulator",
                "output_head_seconds": "native vocabulary projection wall accumulator",
                "wait_seconds": "native placeholder zero; not an isolated wait measurement"}
            self._profile = profile
            self._pending_done = (line, body)
            return profile_line, None
        return line, body

    def turn(self, *args, **kwargs):
        self._profile = self._done_stats = None
        result = super().turn(*args, **kwargs)
        if self._profile is None or self._done_stats is None:
            self.close()
            raise ValueError("missing OLMoE request profile")
        result["engine_request_tokens_per_second"] = result.pop("engine_decode_tokens_per_second")
        result.update(self._done_stats)
        result["protocol_profile"] = self._profile
        result["isolated_prefill_and_decode_available"] = False
        return result


def pilot_statistics(lines):
    result = {}
    for line in lines:
        if not line.startswith("[olmoe PILOT]"): continue
        match = re.fullmatch(r"\[olmoe PILOT\] turn (\S+): queued=(\d+) completed=(\d+) loads=(\d+)\s*", line)
        if not match or match[1] in result: raise ValueError("invalid or duplicate PILOT telemetry")
        values = dict(zip(("queued", "completed", "loads"), (int(match[i]) for i in (2, 3, 4))))
        for value in values.values(): bounded_int(value, 0, 2**32-1, "PILOT count")
        if values["queued"] != values["completed"] or values["loads"] > values["completed"]:
            raise ValueError("request left unfinished or inconsistent prefetch work")
        result[match[1]] = values
    return result


def summary(records):
    def cells(rows):
        ratios = [r["candidate"]["engine_request_tokens_per_second"] / r["control"]["engine_request_tokens_per_second"] for r in rows]
        return {"pairs": len(rows), "median_paired_request_speed_ratio": statistics.median(ratios),
                "paired_ratio_min": min(ratios), "paired_ratio_max": max(ratios),
                "control_median_request_tokens_per_second": statistics.median(r["control"]["engine_request_tokens_per_second"] for r in rows),
                "candidate_median_request_tokens_per_second": statistics.median(r["candidate"]["engine_request_tokens_per_second"] for r in rows),
                "median_paired_wall_speed_ratio": statistics.median(r["control"]["wall_seconds"] / r["candidate"]["wall_seconds"] for r in rows)}
    return {**cells(records), "per_prompt": [{"prompt": i, **cells([r for r in records if r["prompt"] == i])} for i in range(3)]}


def main(argv=None, *, engine_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "model", "prompts", "output"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--wsl")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-per-layer", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    bounded_int(args.threads, 1, 64, "threads"); bounded_int(args.repeats, 5, 100, "repeats")
    bounded_int(args.cache_per_layer, 1, 64, "expert cache")
    bounded_number(args.request_timeout, 0, 600, "request timeout", True)
    args.engine = args.engine.resolve(); args.model = args.model.resolve()
    args.prompts = args.prompts.resolve(); args.output = args.output.resolve()
    if not args.engine.is_file() or not args.model.is_dir(): raise ValueError("engine/model absent")
    if os.path.lexists(args.output): raise FileExistsError("report must be new")
    if args.output.is_relative_to(args.model): raise ValueError("report must be outside model assets")
    if args.prompts.stat().st_size > 256*1024: raise ValueError("prompt file too large")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) != 3 or not all(isinstance(p, str) and p and len(p.encode("utf-8")) <= 65536 for p in prompts) or len(set(prompts)) != 3:
        raise ValueError("provide three distinct bounded prompt strings")
    cfg = json.loads((args.model/"config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "olmoe": raise ValueError("this collector requires native OLMoE")
    common = {"COLI_CUDA":"0", "COLI_VULKAN":"0", "COLI_LOOKUP":"0", "COLI_STRATA_HEADS":"",
              "Q36_MTP":"", "COLI_USAGE":"", "USAGE_SAVE":"0", "COLI_TIMERS":"1",
              "IDOT":"0", "FUSED3":"0", "HOT":"0", "WIDE":"1", "SMOOTH":"0.3",
              "CONF_LIMIT":"0.92", "PILOT_EVICT_GUARD":"1", "COLI_VK_TIER_GB":"0",
              "SERVE_BATCH":"1", "KV_SLOTS":"1", "COLI_KV_PREFIX":"0", "EXPERT_DROP":"0"}
    report = {"schema":"colibri.native-moe-prefetch-benchmark/v1", "status":"running",
              "settings":{"threads":args.threads, "cache_per_layer":args.cache_per_layer, "expert_bits":8,
                          "repeats":args.repeats, "tokens":64, "common_env":common},
              "metric_scope":"whole-request completion rate: includes prefill, decode, and unused prefetch drain",
              "cache_policy":"warm OS cache after asset hashing; no cold SSD/NVMe qualification",
              "correctness":[], "warmups":[], "records":[], "arms":{}}
    engines = {}; failure = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_stream = args.output.open("x", encoding="utf-8")
    def checkpoint():
        output_stream.seek(0)
        json.dump(safe_json(report), output_stream, indent=2, allow_nan=False)
        output_stream.write("\n"); output_stream.truncate(); output_stream.flush()
    try:
        report["binary_sha256"] = file_sha(args.engine)
        report["model_identity"] = model_identity(args.model)
        report["prompt_sha256"] = file_sha(args.prompts)
        checkpoint()
        for side, pilot in (("control", 0), ("candidate", 1)):
            env = dict(common, PILOT=str(pilot))
            engine = (engine_factory or MoeMetricEngine)(args.engine, args.model, args.threads, env, args.wsl,
                           cache_per_layer=args.cache_per_layer, expert_bits=8)
            engines[side] = engine
            report["arms"][side] = {"pilot":pilot, "launch_settings":engine.launch_settings,
                                     "startup_ready_seconds":engine.startup_ready_seconds}
            checkpoint()
        for index, prompt in enumerate(prompts):
            pair = {side:engine.turn(prompt, 64, True, timeout=args.request_timeout) for side,engine in engines.items()}
            equal = check_numeric(pair["control"], pair["candidate"])
            report["correctness"].append({"prompt":index, "equal":equal, **pair}); checkpoint()
            if not equal: raise RuntimeError("prefetch changed token IDs/logprobs/output")
            expected = pair["control"]
            warm = {side:engine.turn(prompt, 64, False, timeout=args.request_timeout) for side,engine in engines.items()}
            report["warmups"].append({"prompt":index, **warm}); checkpoint()
            if not all(equal_output(value, expected) for value in warm.values()): raise RuntimeError("warmup output changed")
            for repeat in range(args.repeats):
                order = ("control", "candidate") if (index+repeat)%2 == 0 else ("candidate", "control")
                pair = {side:engines[side].turn(prompt, 64, False, timeout=args.request_timeout) for side in order}
                report["records"].append({"prompt":index, "repeat":repeat, "order":list(order), **pair}); checkpoint()
                for value in pair.values():
                    bounded_int(value["tokens"], 1, 64, "completed tokens")
                    bounded_number(value["engine_request_tokens_per_second"], 0, 1e9, "request rate", True)
                    bounded_number(value["wall_seconds"], 0, 600, "request wall", True)
                    if not equal_output(value, expected): raise RuntimeError("timed output changed")
                print(json.dumps({"prompt":index, "repeat":repeat, **{side+"_request_tps":pair[side]["engine_request_tokens_per_second"] for side in order}}), flush=True)
        report["summary"] = summary(report["records"])
        if report["binary_sha256"] != file_sha(args.engine) or report["model_identity"] != model_identity(args.model) or report["prompt_sha256"] != file_sha(args.prompts):
            raise RuntimeError("benchmark asset identity changed")
        report["assets_verified_before_and_after"] = True
        report["status"] = "complete"
    except BaseException as error:
        failure = error; report["status"] = "failed"
        report["error"] = {"type":type(error).__name__, "message":str(error)}
    finally:
        for side, engine in engines.items():
            try:
                engine.close()
                report["arms"][side]["stderr"] = list(engine.stderr)
                stats = pilot_statistics(engine.stderr)
                observed = []
                for rows in (report["correctness"], report["warmups"], report["records"]):
                    for row in rows:
                        value = row[side]; evidence = stats.get(value["request_id"])
                        if evidence is None: raise RuntimeError("missing actual PILOT execution evidence")
                        value["native_prefetch_statistics"] = evidence; observed.append(evidence)
                if side == "control" and any(any(cell.values()) for cell in observed): raise RuntimeError("PILOT-off executed prefetch")
                if side == "candidate" and report["status"] == "complete" and not any(cell["loads"] > 0 for cell in observed):
                    raise RuntimeError("PILOT-on did not load experts in its worker")
            except BaseException as error:
                failure = failure or error; report["status"] = "failed"
                report.setdefault("qualification_errors", []).append({"arm":side, "type":type(error).__name__, "message":str(error)})
        try: checkpoint()
        finally: output_stream.close()
    if failure is not None: raise failure


if __name__ == "__main__": main()

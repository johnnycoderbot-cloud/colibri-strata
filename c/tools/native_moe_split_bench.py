#!/usr/bin/env python3
"""Paired OLMoE Vulkan cold-expert split qualification; downloads nothing.

Run this collector directly with the host's Python (including WSL Python).
DONE measures the whole request, including prefill; every request carries the
same bounded token/logprob evidence. Software Vulkan is labeled explicitly.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

try:
    from .native_decode_bench import bounded_int, bounded_number, file_sha
    from .native_moe_prefetch_bench import MoeMetricEngine, pilot_statistics
    from .native_optimization_bench import check_numeric, equal_output, model_identity, safe_json
except ImportError:
    from native_decode_bench import bounded_int, bounded_number, file_sha
    from native_moe_prefetch_bench import MoeMetricEngine, pilot_statistics
    from native_optimization_bench import check_numeric, equal_output, model_identity, safe_json

ROOT = Path(__file__).resolve().parents[1]
SPLIT = re.compile(r" \| split: (native|mojo) (\d+) decisions, (?:planned )?cold device (\d+) CPU (\d+), deferred (\d+), calibration (\d+), predicted ([0-9.]+) ms \((live decode costs|caller profile)\)(?= \||$)")
TIER = re.compile(r"\[VK\] tier olmoe turn: device (\d+) of (\d+) routed experts .*")
DEVICE = re.compile(r"\[VK\] ready: (.+), compute qfam (\d+), memtype (\d+)(?:,.*)?")
SOFTWARE = re.compile(r"llvmpipe|lavapipe|swiftshader|softpipe|software|microsoft basic render|cpu|\bWARP\b", re.I)


def asset_manifest(root, suffixes, *, shader_sources=False):
    paths = [p for p in root.iterdir() if p.suffix.lower() in suffixes]
    if shader_sources and (root / "shaders").is_dir():
        paths += [p for p in (root / "shaders").iterdir() if p.suffix.lower() == ".comp"]
    if not paths or len(paths) > 4096:
        raise ValueError("empty or excessive source/shader asset set")
    files = []
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("asset must be a regular non-symlink file")
        before = path.stat(); digest = file_sha(path); after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("asset changed during hashing")
        files.append({"name": path.relative_to(root).as_posix(), "bytes": before.st_size, "sha256": digest})
    return {"path": str(root), "files": files,
            "manifest_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def identities(args):
    return {"binary": {"path": str(args.engine), "sha256": file_sha(args.engine)},
            "collector_sha256":file_sha(Path(__file__)),
            "model": model_identity(args.model), "prompts_sha256": file_sha(args.prompts),
            "source": asset_manifest(args.source_root, {".c", ".h", ".mojo"}, shader_sources=True),
            "shaders": asset_manifest(args.shaders, {".spv"}),
            "mojo_library": {"path": str(args.mojo_library), "sha256": file_sha(args.mojo_library)} if args.mojo_library else None,
            "vulkan_icd": {"path": str(args.vulkan_icd), "sha256": file_sha(args.vulkan_icd)} if args.vulkan_icd else None}


@contextmanager
def isolated_selection():
    # NativeEngine inherits this process's environment. Strip overrides by
    # name without retaining or reporting potentially private ambient values.
    prefixes = ("COLI_VK", "COLI_MOJO", "COLI_CUDA", "COLI_VULKAN", "CUDA_", "VK_")
    saved = {key: value for key, value in os.environ.items() if key.startswith(prefixes)}
    try:
        for key in saved: os.environ.pop(key, None)
        yield
    finally:
        for key in list(os.environ):
            if key.startswith(prefixes): os.environ.pop(key, None)
        os.environ.update(saved)


def numeric_comparison(control, candidate, tolerance):
    if not equal_output(control, candidate) or control.get("text") != candidate.get("text"):
        raise RuntimeError("token count/text/output bytes differ")
    if (control.get("token_ids_verified") is not True or candidate.get("token_ids_verified") is not True or
            control.get("token_ids") != candidate.get("token_ids")):
        raise RuntimeError("exact greedy token IDs differ or are unverified")
    n = bounded_int(control["tokens"], 1, 64, "completed tokens")
    if len(control["token_ids"]) != n:
        raise ValueError("token ID count differs")
    left, right = control.get("numeric_evidence"), candidate.get("numeric_evidence")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != n or len(right) != n:
        raise ValueError("missing numeric token evidence")
    error = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if any(cell.get("emitted_index") != index or cell.get("token_id") != control["token_ids"][index] or
               cell.get("token_id_source") != "top1_greedy" for cell in (a, b)):
            raise ValueError("inconsistent token evidence identity")
        if a.get("body_sha256") != b.get("body_sha256"):
            raise RuntimeError("token body bytes differ")
        for cell in (a, b): bounded_number(cell.get("logprob"), -1e6, 1e6, "chosen logprob")
        error = max(error, abs(a["logprob"] - b["logprob"]))
        at, bt = a.get("top_tokens"), b.get("top_tokens")
        if not isinstance(at, list) or not isinstance(bt, list) or not 1 <= len(at) <= 20 or len(at) != len(bt):
            raise ValueError("invalid top-token evidence")
        if [cell["token_id"] for cell in at] != [cell["token_id"] for cell in bt] or at[0]["token_id"] != a["token_id"]:
            raise RuntimeError("top-token identities differ")
        for x, y in zip(at, bt):
            bounded_number(x["logprob"], -1e6, 1e6, "top logprob")
            bounded_number(y["logprob"], -1e6, 1e6, "top logprob")
            error = max(error, abs(x["logprob"] - y["logprob"]))
    if error > tolerance:
        raise RuntimeError(f"maximum absolute logprob error {error:.9g} exceeds {tolerance:.9g}")
    return {"exact_token_ids_counts_text": True, "max_absolute_logprob_error": error,
            "absolute_logprob_tolerance": tolerance, "exact_numeric_frames": check_numeric(control, candidate)}


def tier_statistics(lines):
    rows = []; previous = None
    names = ("decisions", "planned_cold_device", "planned_cold_host", "deferred", "calibration", "predicted_ms")
    for line in lines:
        line = line.strip()
        if not line.startswith("[VK] tier olmoe turn:"): continue
        match = TIER.fullmatch(line)
        if not match: raise ValueError("malformed Vulkan tier telemetry")
        routed = bounded_int(int(match[2]), 0, 2**63-1, "routed experts")
        served = bounded_int(int(match[1]), 0, routed, "device experts")
        row = {"device_experts_cumulative": served, "routed_experts_cumulative": routed}
        split = SPLIT.search(line)
        if " | split:" in line and split is None: raise ValueError("malformed split telemetry")
        if split:
            cumulative = {name: bounded_int(int(split[i+2]), 0, 2**63-1, name) for i, name in enumerate(names[:-1])}
            cumulative["predicted_ms"] = bounded_number(float(split[7]), 0, 1e15, "predicted timing")
            if previous is not None and any(cumulative[key] < previous[key] for key in names):
                raise ValueError("split cumulative telemetry decreased")
            row.update(split_mode=split[1], profile_source=split[8], split_cumulative=cumulative,
                       split_since_previous_turn={key: cumulative[key] - (previous[key] if previous else 0) for key in names})
            previous = cumulative
        rows.append(row)
    return rows


def device_evidence(lines):
    found = []
    for line in lines:
        if not line.startswith("[VK] ready:"): continue
        match = DEVICE.fullmatch(line.strip())
        if not match: raise ValueError("malformed Vulkan device READY")
        found.append({"device_name": match[1], "compute_queue_family": int(match[2]), "memory_type": int(match[3]),
                      "software_device": bool(SOFTWARE.search(match[1])), "source": "captured native Vulkan deviceName"})
    if len(found) != 1: raise RuntimeError("missing or ambiguous actual Vulkan device evidence")
    found[0]["scope"] = "software Vulkan; no GPU-speed claim" if found[0]["software_device"] else "reported Vulkan device; physical type not independently verified"
    return found[0]


def summary(records):
    def cells(rows):
        ratios = [r["candidate"]["engine_request_tokens_per_second"] / r["control"]["engine_request_tokens_per_second"] for r in rows]
        return {"pairs": len(rows), "median_paired_whole_request_speed_ratio": statistics.median(ratios),
                "paired_ratio_min": min(ratios), "paired_ratio_max": max(ratios),
                "control_median_request_tokens_per_second": statistics.median(r["control"]["engine_request_tokens_per_second"] for r in rows),
                "candidate_median_request_tokens_per_second": statistics.median(r["candidate"]["engine_request_tokens_per_second"] for r in rows),
                "median_paired_wall_speed_ratio": statistics.median(r["control"]["wall_seconds"] / r["candidate"]["wall_seconds"] for r in rows),
                "candidate_wins": sum(ratio > 1 for ratio in ratios)}
    return {**cells(records), "per_prompt": [{"prompt": i, **cells([r for r in records if r["prompt"] == i])} for i in range(3)]}


def main(argv=None, *, engine_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "model", "prompts", "output"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--candidate", choices=("native", "mojo"), default="mojo")
    parser.add_argument("--mojo-library", type=Path)
    parser.add_argument("--shaders", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-per-layer", type=int, default=8)
    parser.add_argument("--tier-gb", type=float, default=.06)
    parser.add_argument("--tier-policy", choices=("lru", "lfru"), default="lru")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-logprob-error", type=float, default=.02)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args(argv)
    bounded_int(args.threads,1,64,"threads");bounded_int(args.tokens,8,64,"tokens");bounded_int(args.repeats,3,20,"repeats")
    bounded_int(args.device_index,0,64,"device index")
    if args.cache_per_layer != 8: raise ValueError("paired qualification fixes RAM expert cache at 8 per layer")
    bounded_number(args.tier_gb,0,4,"tier GiB",True);bounded_number(args.max_logprob_error,0,.1,"logprob tolerance")
    for key in ("request_timeout","startup_timeout"): bounded_number(getattr(args,key),0,600,key,True)
    for key in ("engine","model","prompts","output","source_root"):
        setattr(args,key,getattr(args,key).resolve())
    if not args.engine.is_file() or not args.model.is_dir() or not args.source_root.is_dir(): raise ValueError("engine/model/source absent")
    if args.candidate == "mojo" and not args.mojo_library: raise ValueError("Mojo candidate requires explicit --mojo-library")
    if args.candidate != "mojo" and args.mojo_library: raise ValueError("native candidate does not load a Mojo library")
    for key in ("mojo_library","vulkan_icd"):
        if getattr(args,key):
            path=getattr(args,key).resolve()
            if not path.is_file(): raise ValueError(key+" must exist")
            setattr(args,key,path)
    args.shaders=(args.shaders or (args.engine.parent/"shaders" if (args.engine.parent/"shaders/qmatmul.spv").is_file() else ROOT/"shaders")).resolve()
    if not (args.shaders/"qmatmul.spv").is_file(): raise ValueError("explicit shader set requires qmatmul.spv")
    if os.path.lexists(args.output): raise FileExistsError("report must be a new path")
    if args.output.is_relative_to(args.model) or args.output in (args.engine,args.prompts,args.mojo_library,args.vulkan_icd):
        raise ValueError("report must be outside input assets")
    if args.prompts.stat().st_size > 256*1024: raise ValueError("prompt file exceeds 256 KiB")
    prompts=json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts,list) or len(prompts)!=3 or not all(isinstance(p,str) and p and len(p.encode("utf-8","strict"))<=65536 for p in prompts) or len(set(prompts))!=3:
        raise ValueError("provide exactly three distinct bounded prompts")
    if json.loads((args.model/"config.json").read_text(encoding="utf-8")).get("model_type") != "olmoe": raise ValueError("collector requires native OLMoE")
    common={"COLI_CUDA":"0","COLI_VULKAN":"1","COLI_LOOKUP":"0","COLI_STRATA_HEADS":"","Q36_MTP":"",
            "COLI_USAGE":"","USAGE_SAVE":"0","COLI_TIMERS":"1","IDOT":"0","FUSED3":"0","PILOT":"0",
            "HOT":"0","WIDE":"1","SMOOTH":"0.3","CONF_LIMIT":"0.92","PILOT_EVICT_GUARD":"1",
            "COLI_VK_TIER":"1","COLI_VK_TIER_GB":str(args.tier_gb),"COLI_VK_TIER_POLICY":args.tier_policy,
            "COLI_VK_TIER_STREAM":"1","COLI_VK_TIER_WARM":"0","COLI_VK_TIER_SYNC":"1","COLI_VK_DENSE":"0","COLI_VK_CHAIN":"0",
            "COLI_VK_DEV":str(args.device_index),"COLI_VK_SHADERS":str(args.shaders),"SERVE_BATCH":"1",
            "KV_SLOTS":"1","COLI_KV_PREFIX":"0","EXPERT_DROP":"0","COLI_DENSE_IDOT":"0"}
    if args.vulkan_icd: common["VK_ICD_FILENAMES"]=str(args.vulkan_icd)
    report={"schema":"colibri.native-moe-split-benchmark/v1","status":"running","settings":{
            "threads":args.threads,"tokens":args.tokens,"repeats":args.repeats,"expert_bits":8,"cache_per_layer":8,
            "tier_gb":args.tier_gb,"tier_policy":args.tier_policy,"candidate":args.candidate,"common_env":common,
            "max_absolute_logprob_error":args.max_logprob_error,"tier_sync":1},"correctness":[],"warmups":[],"records":[],"arms":{},
            "metric_scope":"whole-request DONE rate: prefill + decode + request-owned drain; numeric channel on both arms",
            "isolated_decode_metric_available":False,"physical_gpu_execution_qualified":False,"gpu_speed_claim_allowed":False,
            "cache_scope":"asset hashing warms the shared OS page cache; not a cold storage measurement",
            "source_scope":"current declared native sources hashed; binary build provenance is not inferred",
            "split_counter_scope":"planned cold work; actual total Vulkan expert execution is reported separately",
            "tier_sync_scope":"qualification forces SYNC=1 so promotions drain at quiescent joins; ordinary SYNC=0 remains available",
            "held_out_scope":"caller-provided three distinct prompts; training disjointness not inferred",
            "fake_engine_factory":engine_factory is not None,
            "invocation":{"executable":sys.executable,"arguments":list(argv) if argv is not None else sys.argv[1:]}}
    engines={};requests={"control":[],"candidate":[]};failure=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as output:
        def checkpoint():
            output.seek(0);json.dump(safe_json(report),output,indent=2,allow_nan=False);output.write("\n");output.truncate();output.flush()
        try:
            report["identities_before"]=identities(args);checkpoint()
            for side,mode in (("control","off"),("candidate",args.candidate)):
                env=dict(common,COLI_VK_TIER_SPLIT=mode)
                if mode=="mojo": env["COLI_MOJO_DISPATCH"]=str(args.mojo_library)
                with isolated_selection():
                    engine=(engine_factory or MoeMetricEngine)(args.engine,args.model,args.threads,env,None,
                        cache_per_layer=8,expert_bits=8,startup_timeout=args.startup_timeout)
                engines[side]=engine
                report["arms"][side]={"launch_settings":engine.launch_settings,"startup_ready_seconds":engine.startup_ready_seconds,
                    "ready_line":getattr(engine,"ready_line",None)};checkpoint()
            def paired(prompt,index,section,repeat=None):
                order=("control","candidate") if (index+(repeat or 0))%2==0 else ("candidate","control")
                row={"prompt":index,"order":list(order)}
                if repeat is not None: row["repeat"]=repeat
                report[section].append(row)
                for side in order:
                    result=engines[side].turn(prompt,args.tokens,True,timeout=args.request_timeout)
                    row[side]=result;requests[side].append(result);checkpoint()
                    bounded_int(result["tokens"],1,args.tokens,"completed tokens")
                    bounded_number(result["engine_request_tokens_per_second"],0,1e9,"request rate",True)
                    bounded_number(result["wall_seconds"],0,600,"request wall",True)
                row["numeric_comparison"]=numeric_comparison(row["control"],row["candidate"],args.max_logprob_error);checkpoint()
                return row
            for index,prompt in enumerate(prompts):
                expected=paired(prompt,index,"correctness")["control"]
                for section in ("warmups","records"):
                    for repeat in range(1 if section=="warmups" else args.repeats):
                        row=paired(prompt,index,section,repeat if section=="records" else None)
                        for side in engines:
                            numeric_comparison(expected,row[side],args.max_logprob_error)
                        if section=="records": print(json.dumps({"prompt":index,"repeat":repeat,
                            "control_request_tps":row["control"]["engine_request_tokens_per_second"],
                            "candidate_request_tps":row["candidate"]["engine_request_tokens_per_second"]}),flush=True)
            report["summary"]=summary(report["records"])
            report["numeric_evidence_summary"]={"exact_token_ids_counts_text":True,"absolute_logprob_tolerance":args.max_logprob_error,
                "maximum_observed_absolute_logprob_error":max(row["numeric_comparison"]["max_absolute_logprob_error"]
                    for section in ("correctness","warmups","records") for row in report[section])}
            report["status"]="complete"
        except BaseException as error:
            failure=error;report["status"]="failed";report["error"]={"type":type(error).__name__,"message":str(error)}
        finally:
            for side,engine in engines.items():
                try:
                    engine.close()
                    process=getattr(engine,"process",None)
                    if process is not None and process.poll() is None: raise RuntimeError("benchmark child did not exit")
                except BaseException as error:
                    failure=failure or error;report["status"]="failed"
                    report.setdefault("cleanup_errors",[]).append({"arm":side,"type":type(error).__name__,"message":str(error)})
                arm=report["arms"][side];arm["stderr"]=list(engine.stderr)
                arm["stderr_truncated_bytes"]=getattr(engine,"stderr_truncated_bytes",0)
                try:
                    if arm["stderr_truncated_bytes"]: raise RuntimeError("execution telemetry was truncated")
                    arm["device"]=device_evidence(engine.stderr)
                    stats=tier_statistics(engine.stderr)
                    if len(stats)!=len(requests[side]): raise RuntimeError("missing or ambiguous per-turn Vulkan telemetry")
                    for result,cell in zip(requests[side],stats):
                        result["native_tier_statistics"]=cell
                        if side=="control" and "split_mode" in cell: raise RuntimeError("control executed split policy")
                        if side=="candidate" and cell.get("split_mode")!=args.candidate: raise RuntimeError("candidate split policy not executed")
                        if side=="candidate" and cell.get("profile_source")!="live decode costs": raise RuntimeError("candidate did not use achieved live decode costs")
                    pilot=pilot_statistics(engine.stderr)
                    if set(pilot)!={value["request_id"] for value in requests[side]}: raise RuntimeError("missing actual CPU-prefetch-off telemetry")
                    if any(any(cell.values()) for cell in pilot.values()): raise RuntimeError("disabled CPU prefetch executed work")
                    if report["status"]=="complete" and (not stats or stats[-1]["device_experts_cumulative"]==0): raise RuntimeError("no expert execution on Vulkan device")
                    if side=="candidate" and report["status"]=="complete" and not any(cell["split_since_previous_turn"]["decisions"]>0 for cell in stats): raise RuntimeError("candidate made no split decisions")
                except BaseException as error:
                    failure=failure or error;report["status"]="failed"
                    report.setdefault("qualification_errors",[]).append({"arm":side,"type":type(error).__name__,"message":str(error)})
            try:
                report["identities_after"]=identities(args)
                if report.get("identities_before")!=report["identities_after"]: raise RuntimeError("benchmark input assets changed")
                report["assets_verified_before_and_after"]=True
                devices=[arm.get("device") for arm in report["arms"].values()]
                if len(devices)==2 and all(devices) and devices[0]!=devices[1]: raise RuntimeError("arms executed on different Vulkan devices")
                report["execution_scope"]="software Vulkan; no GPU-speed claim" if any(d and d["software_device"] for d in devices) else "native Vulkan split comparison; physical GPU type remains unqualified"
            except BaseException as error:
                failure=failure or error;report["status"]="failed"
                report.setdefault("qualification_errors",[]).append({"type":type(error).__name__,"message":str(error)})
            checkpoint()
    if failure is not None: raise failure
    return report


if __name__=="__main__": main()

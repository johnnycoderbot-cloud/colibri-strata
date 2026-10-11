#!/usr/bin/env python3
"""Real native backend qualification, using offline prepared fixtures.

Inference subprocesses contain no Python model bindings. Preparation may use
the SDK's Python exporter. The resulting report is functional evidence, not a
Colibri model-speed or bare-metal OS claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, default=ROOT / "mojo_runtime")
    p.add_argument("--fixtures", type=Path, required=True, help="directory containing f32/i4/i8/i16 packs")
    p.add_argument("--output", type=Path, required=True, help="fresh report/results directory")
    args = p.parse_args()
    if args.output.exists():
        p.error("output must be new")
    args.output.mkdir(parents=True)
    binary = args.native.resolve()
    probe = subprocess.run([str(binary), "--probe"], capture_output=True, check=True, timeout=30)
    report = {"schema": "colibri.native-mojo-qualification/v1",
              "native_binary_sha256": digest(binary),
              "host": {"os": platform.system(), "release": platform.release(), "architecture": platform.machine(),
                       "wsl": "microsoft" in platform.release().lower()},
              "probe": json.loads(probe.stdout), "runs": [], "accelerator_refusal": None,
              "scope": "functional synthetic graphs, not full-model throughput or universal hardware support"}
    kinds = ["f32", "i4", "i8", "i16"]
    if (args.fixtures / "weighted").is_dir():
        kinds.append("weighted")
    for kind in kinds:
        fixture = args.fixtures / kind
        target = args.output / kind
        before = time.perf_counter()
        command = [str(binary), "--model", str((fixture/"model.mef").resolve()),
                                "--inputs", str((fixture/"inputs.json").resolve()),
                                "--output", str(target.resolve()), "--device", "CPU", "--repeat", "5"]
        if kind == "weighted":
            command += ["--weights", str((fixture/"weights.safetensors").resolve())]
        child = subprocess.run(command,
                               capture_output=True, timeout=180, check=True)
        wall = time.perf_counter()-before
        actual, expected = (target/"output-0.bin").read_bytes(), (fixture/"expected.bin").read_bytes()
        if actual != expected:
            raise AssertionError(kind + " differs from independent fixture reference")
        result = json.loads(child.stdout)
        assert result["python_model_bindings"] is False
        report["runs"].append({"fixture": kind, "exact_reference_match": True, "external_wall_seconds": wall,
                               "model_sha256": digest(fixture/"model.mef"), "output_sha256": digest(target/"output-0.bin"),
                               "result": result})
    if report["probe"]["accelerator_count"] == 0:
        fixture = args.fixtures/"f32"
        child = subprocess.run([str(binary), "--model", str((fixture/"model.mef").resolve()),
                                "--inputs", str((fixture/"inputs.json").resolve()), "--output", str(args.output/"absent"),
                                "--device", "ACCELERATOR.0"], capture_output=True, timeout=30)
        assert child.returncode != 0 and not child.stdout and not (args.output/"absent").exists()
        report["accelerator_refusal"] = {"passed": True, "cpu_fallback": False,
                                         "diagnostic": child.stderr.decode("utf-8", "replace").strip()}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"native_cpu_fixture_runs": len(kinds), "exact": True,
                      "wsl": report["host"]["wsl"], "accelerators": report["probe"]["accelerator_count"]}))


if __name__ == "__main__":
    main()

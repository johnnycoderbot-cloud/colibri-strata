#!/usr/bin/env python3
"""Run an existing, compiled Metis artifact using an explicit installed SDK.

This verifies the supplied vendor artifact, not arbitrary Colibri model support.
No compilation, driver installation, reset, firmware flash or power changes.
"""
import argparse
import json
import os
import hashlib
import re
from pathlib import Path
import subprocess

ENTRY = 'import sys; sys.argv=["axrunmodel",*sys.argv[1:]]; from axelera.runtime.run import entrypoint_main; sys.exit(entrypoint_main())'


def invoke(sdk_python, runtime_bin, device_dir, arguments, timeout=30):
    for value in (sdk_python, runtime_bin, device_dir):
        if not Path(value).is_absolute() or not Path(value).exists():
            raise ValueError("explicit existing SDK paths are required")
    env = os.environ.copy()
    env["PATH"] = str(runtime_bin) + os.pathsep + str(Path(runtime_bin).parent / "lib") + os.pathsep + env.get("PATH", "")
    env["AXELERA_DEVICE_DIR"] = str(device_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([str(sdk_python), "-X", "utf8", "-c", ENTRY, *arguments],
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env)


def verification_counts(stdout, return_code):
    if return_code != 0 or re.search(r"validation:\s*(?:fail|error)", stdout, re.I):
        return False, 0, 0
    all_frames = re.findall(r"validation:\s*ok\s+validated all (\d+) checked frames", stdout)
    summaries = re.findall(r"validation:\s*ok\s+validated\s+([\d.]+)%\s+of\s+(\d+)\s+frames\s*\(checked=(\d+)\s+not_validated=(\d+)\)", stdout)
    if len(all_frames) == 1 and not summaries:
        total = int(all_frames[0])
        return total > 0, total, total
    if len(summaries) == 1 and not all_frames:
        _, total, checked, skipped = summaries[0]
        total, checked, skipped = int(total), int(checked), int(skipped)
        return checked > 0 and checked == total and skipped == 0, checked, total
    return False, 0, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-python", type=Path, required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--device-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--help-sdk", action="store_true")
    parser.add_argument("--seconds", type=int, default=2)
    args = parser.parse_args()
    if args.help_sdk:
        result = invoke(args.sdk_python, args.runtime_bin, args.device_dir, ["--help"])
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)
    if not args.model or not args.model.is_file() or not args.out or args.out.exists() or args.out.with_suffix(".stderr.log").exists():
        parser.error("provide an existing compiled model.json and a new report filename")
    if not 1 <= args.seconds <= 10:
        parser.error("seconds must be 1..10")
    result = invoke(args.sdk_python, args.runtime_bin, args.device_dir,
                    [str(args.model.resolve()), "--seconds", str(args.seconds), "--verify", "every", "--stats"], timeout=60)
    passed, checked, total = verification_counts(result.stdout, result.returncode)
    report = {"schema": "colibri.strata.axelera-compiled-probe/v1", "passed": bool(passed),
              "return_code": result.returncode, "checked_frames": checked, "total_frames": total,
              "model_descriptor_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "model": args.model.parent.parent.name, "scope": "existing vendor-compiled artifact",
              "colibri_head_integration_tested": False, "native_int4_int16_tested": False,
              "seconds": args.seconds, "stdout": result.stdout,
              "stderr_informational_present": bool(result.stderr)}
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if not passed:
        # Keep detailed diagnostics locally; SDK stderr can contain device IDs.
        with args.out.with_suffix(".stderr.log").open("x", encoding="utf-8") as stream:
            stream.write(result.stderr)
        print(result.stderr[-2000:])
    print(json.dumps({key: report[key] for key in ("passed", "return_code", "checked_frames", "total_frames", "scope")}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

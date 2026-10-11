#!/usr/bin/env python3
"""Replay prepared packs through production HTTP routes with Python ML blocked.

This is functional native-port qualification, not a throughput benchmark.
Supply real immutable model packs and replay fixtures; no capture or download.
"""
import argparse
import builtins
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("text", "speech", "camera", "audio", "image", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["CPU", "GPU"])
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    forbidden = {"openvino", "openvino_genai", "openvino_tokenizers",
                 "numpy", "PIL", "torch", "jax", "onnxruntime"}
    original = builtins.__import__

    def guarded(name, *values, **keywords):
        if name.split(".")[0] in forbidden:
            raise AssertionError("Python model/tensor import attempted: " + name)
        return original(name, *values, **keywords)

    builtins.__import__ = guarded
    try:
        from check_runtime_gateway import check
        from native_graph import native_binary
        runs = []
        for device in args.devices:
            for name, fixture in (("text", None), ("speech", args.audio), ("camera", args.image)):
                report = check(str(getattr(args, name)), device, fixture)
                assert report["runtime"]["inference_runtime"] == "native-c"
                assert report["http_status"] == 200
                report["workload"] = name
                # check returns only after its GraphEngine context closes.
                report["native_process_close_passed"] = True
                # The runtime does not contain input/output file paths. Keep the
                # evidence portable and avoid publishing private machine paths.
                runs.append(report)
                print(json.dumps({"workload": name, "device": device,
                                  "status": 200, "runtime": "native-c"}), flush=True)
        loaded = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
        assert not loaded, loaded
        evidence = {"schema": "colibri.native-graph-functional-replay/v1",
                    "purpose": "functional C production routes; no throughput/power claim",
                    "native_binary_sha256": digest(native_binary()),
                    "python_ml_import_guard": sorted(forbidden),
                    "python_ml_modules_loaded": loaded,
                    "fixtures": {"audio_sha256": digest(args.audio),
                                 "image_sha256": digest(args.image)},
                    "model_manifest_sha256": {name: digest(getattr(args, name) / "colibri_runtime.json")
                                              for name in ("text", "speech", "camera")},
                    "runs": runs}
        with args.report.open("x", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
            f.write("\n")
    finally:
        builtins.__import__ = original


if __name__ == "__main__":
    main()

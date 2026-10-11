#!/usr/bin/env python3
"""Explicit accelerator and regular-file integrity checks; never claims CPU is GPU.

Requires a synthetic/real head pack for GPU/CPU or JAX checks. No raw disk writes,
driver/firmware changes, model downloads, or cloud TPU allocation is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import heads
import precision
from storage import FileTier, ReadRequest


def storage_check(directory, reads=64):
    if type(reads) is not int or not 1 <= reads <= 4096:
        raise ValueError("reads must be 1..4096")
    directory = Path(directory).resolve()
    rng = random.Random(20261010)
    data = rng.randbytes(8 * 1024 * 1024)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tmp_strata_storage_", dir=directory) as temp:
        path = Path(temp) / "fixture.bin"
        with path.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        with FileTier(path) as tier:
            for first in range(0, reads, 2):
                pending = []
                for index in range(first, min(first + 2, reads)):
                    offset = rng.randrange(0, len(data) - 1024 * 1024)
                    expected = data[offset:offset + 1024 * 1024]
                    request = ReadRequest(f"read-{index}", offset, len(expected), hashlib.sha256(expected).hexdigest())
                    pending.append((tier.submit(request), request, expected))
                for future, request, expected in pending:
                    result = future.result(timeout=30)
                    if result.request != request or result.data != expected:
                        raise ValueError("storage identity/data mismatch")
        # Actual stored CSQ1 objects, not just generic file throughput.
        vectors = [precision.f32((i % 31 - 15) / 3) for i in range(257)]
        blobs = [precision.encode(vectors, bits, 17) for bits in (4, 8, 16)]
        path.write_bytes(b"".join(blobs))
        with FileTier(path) as tier:
            offset = 0
            for bits, blob in zip((4, 8, 16), blobs):
                request = ReadRequest(f"CSQ1-int{bits}", offset, len(blob), hashlib.sha256(blob).hexdigest())
                read = tier.submit(request).result(timeout=30)
                info, values = precision.decode(read.data)
                if info["bits"] != bits or len(values) != len(vectors):
                    raise ValueError("stored quantization artifact mismatch")
                offset += len(blob)
    return {"passed": True, "regular_file": True, "io_mode": "buffered",
            "read_count": reads, "verified_bytes": reads * 1024 * 1024,
            "stored_precisions": [4, 8, 16], "elapsed_seconds": time.perf_counter() - started,
            "physical_nvme_certified": False,
            "note": "Filesystem/device identity must be recorded separately; no O_DIRECT/GPUDirect claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--provider-options", default="{}")
    parser.add_argument("--jax-platform")
    parser.add_argument("--openvino-device", help="explicit CPU/GPU/NPU device; AUTO is not a hardware proof")
    parser.add_argument("--storage-dir", type=Path, default=heads.ROOT)
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.pack and (args.jax_platform or args.openvino_device or args.provider != "CPUExecutionProvider" or args.provider_options != "{}"):
        parser.error("accelerator/provider options require --pack; a storage-only check cannot certify them")
    if args.out.exists():
        parser.error("report output must be a new file")
    report = {"schema": "colibri.strata.hardware-check/v1", "storage": storage_check(args.storage_dir, args.reads)}
    if args.pack:
        import runtime
        import numpy as np
        pack = json.loads(args.pack.read_text(encoding="utf-8"))
        meta = heads.validate_pack(pack)
        options = json.loads(args.provider_options)
        if not isinstance(options, dict):
            parser.error("provider options must be a JSON object")
        hidden = [0.] * meta["hidden_size"]
        hidden[0] = 1.
        runs = []
        for bits in (4, 8, 16):
            expected = np.asarray(precision.reference(pack, hidden, bits, 3))
            result = runtime.run_onnx(pack, hidden, provider=args.provider, provider_options=options,
                                      bits=bits, block_values=3)
            np.testing.assert_allclose(np.asarray(result["logits"]), expected, rtol=2e-5, atol=2e-5)
            if result["tokens"] != expected.argmax(axis=1).tolist():
                raise ValueError("accelerator token parity failed")
            result["passed"] = True
            runs.append(result)
            if args.openvino_device:
                ov_result = runtime.run_openvino(pack, hidden, device=args.openvino_device, bits=bits, block_values=3)
                np.testing.assert_allclose(np.asarray(ov_result["logits"]), expected, rtol=2e-5, atol=2e-5)
                if ov_result["tokens"] != expected.argmax(axis=1).tolist():
                    raise ValueError("OpenVINO token parity failed")
                ov_result["passed"] = True
                report.setdefault("openvino", []).append(ov_result)
        report["onnx"] = runs
        if args.jax_platform:
            report["jax"] = runtime.run_jax(pack, hidden, platform=args.jax_platform)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.out), "storage_passed": report["storage"]["passed"],
                      "provider": args.provider if args.pack else None}))


if __name__ == "__main__":
    main()

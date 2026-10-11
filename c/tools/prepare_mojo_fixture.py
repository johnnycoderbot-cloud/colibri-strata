#!/usr/bin/env python3
"""Offline MAX/Mojo graph fixture preparation; production execution stays C."""
import argparse
import json
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "accelerator"], default="cpu")
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--integer-bits", type=int, choices=[4, 8, 16])
    p.add_argument("--external-weights", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("output must be new; keep prepared packs immutable")
    import numpy as np
    from max.dtype import DType
    from max.driver import CPU, Accelerator, accelerator_count
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType, Weight, ops
    if args.device == "accelerator":
        if args.id < 0 or args.id >= accelerator_count():
            p.error("requested runtime accelerator unavailable; no CPU export fallback")
        device, ref = Accelerator(args.id), DeviceRef.GPU(args.id)
    else:
        device, ref = CPU(), DeviceRef.CPU()
    root = args.output.resolve()
    root.mkdir(parents=True)
    if args.integer_bits:
        bits = args.integer_bits
        values = np.array([-(1 << (bits-1)), -1, 0, 1, (1 << (bits-1))-1], dtype=np.int64)
        if bits == 4:
            payload = bytes([(int(values[j]) & 15) | ((int(values[j+1]) & 15) << 4)
                             if j+1 < len(values) else int(values[j]) & 15 for j in range(0, len(values), 2)])
        else:
            payload = values.astype(np.int8 if bits == 8 else np.int16).tobytes()
        with Graph("colibri_signed_integer", input_types=[TensorType(DType.int8 if bits <= 8 else DType.int16, [5], device=ref)]) as graph:
            value = ops.cast(graph.inputs[0].tensor, DType.int64)
            graph.output(value * ops.constant(np.array(3, dtype=np.int64), device=ref))
        model = InferenceSession(devices=[device]).load(graph)
        model._export_mef(str(root / "model.mef"))
        (root / "input0.bin").write_bytes(payload)
        expected = values * 3
        expected.tofile(root / "expected.bin")
        inputs = {"inputs": [{"name": model.input_metadata[0].name, "dtype": "i"+str(bits), "shape": [5], "file": "input0.bin"}], "outputs": ["output0"]}
        (root / "inputs.json").write_text(json.dumps(inputs, indent=2)+"\n", encoding="utf-8")
        (root / "fixture.json").write_text(json.dumps({"schema":"colibri.mojo-integer-fixture/v1","purpose":"synthetic signed-integer functional fixture; no speed claim","storage_bits":bits,"runtime_input_bits":8 if bits == 4 else bits,"math_bits":64,"expected":expected.tolist()},indent=2)+"\n",encoding="utf-8")
        print(root)
        return
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.float32)
    w = np.array([[2, 1, -1, 3], [4, -2, 2, 1], [-1, 3, 5, -2]], dtype=np.float32)
    types = [TensorType(DType.float32, [2, 3], device=ref)]
    if not args.external_weights:
        types.append(TensorType(DType.float32, [3, 4], device=ref))
    with Graph("colibri_mojo_projection", input_types=types) as graph:
        weight = Weight("projection.weight", DType.float32, [3, 4], device=ref).tensor if args.external_weights else graph.inputs[1].tensor
        graph.output(ops.relu(graph.inputs[0].tensor @ weight))
    session = InferenceSession(devices=[device])
    model = session.load(graph, weights_registry={"projection.weight": w} if args.external_weights else None)
    model._export_mef(str(root / "model.mef"))
    x.tofile(root / "input0.bin")
    w.tofile(root / "input1.bin")
    expected = np.maximum(x @ w, 0)
    expected.tofile(root / "expected.bin")
    inputs = {"inputs": [
        {"name": model.input_metadata[0].name, "dtype": "f32", "shape": [2, 3], "file": "input0.bin"},
    ], "outputs": ["output0"]}
    if args.external_weights:
        raw = w.tobytes()
        header = json.dumps({"projection.weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, len(raw)]}}, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw)
    else:
        inputs["inputs"].append({"name": model.input_metadata[1].name, "dtype": "f32", "shape": [3, 4], "file": "input1.bin"})
    (root / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    (root / "fixture.json").write_text(json.dumps({
        "schema": "colibri.mojo-tensor-fixture/v1",
        "purpose": "synthetic functional projection/relu fixture; no model speed claim",
        "device": args.device, "device_id": args.id, "shape": [2, 4],
        "expected": expected.tolist(), "native_run_input": inputs,
    }, indent=2) + "\n", encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()

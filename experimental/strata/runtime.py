"""Optional accelerator runners for independent trained future-token heads.

CSQ1 INT4/INT8/INT16 storage is decoded into F32 tensors for compatibility.
These graphs perform F32 compute; they are not native integer accelerator
kernels or model-engine integration. ONNX execution is accepted only when a
bounded node profile proves that the requested provider executed the compute.
JAX explicitly places/jits on the requested available platform and verifies
the output device. CPU, GPU, NPU, and TPU availability are distinct outcomes.
"""
import importlib
import json
from pathlib import Path
import tempfile

import heads
import precision

MAX_PROFILE_BYTES = 16 * 1024 * 1024
MAX_PROFILE_EVENTS = 100_000


def _dependency(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"optional dependency {name!r} is unavailable") from exc


def _input(pack, hidden, expected_identity):
    meta = heads.validate_pack(pack, expected_identity)
    np = _dependency("numpy")
    try:
        array = np.asarray(hidden)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hidden must contain finite F32 numbers") from exc
    heads.require(array.dtype.kind in "iuf", "hidden must contain real numbers")
    if array.shape == (meta["hidden_size"],):
        array = array.reshape(1, meta["hidden_size"])
    heads.require(array.shape == (1, meta["hidden_size"]), "hidden shape must be [H] or [1,H]")
    heads.require(bool(np.all(np.isfinite(array))) and bool(np.all(np.abs(array) <= heads.FLOAT32_MAX)),
                  "hidden must contain finite F32 numbers")
    array = np.ascontiguousarray(array, dtype=np.float32)
    heads.require(bool(np.all(np.isfinite(array))), "hidden conversion produced nonfinite F32")
    return np, meta, array


def _weights(pack, bits, block_values):
    # precision.encode validates bit width/block sizes even for zero weights.
    encoded = precision.quantized_weights(pack, bits, block_values)
    return [(precision.decode(row["down"])[1], precision.decode(row["up"])[1]) for row in encoded]


def _metadata(pack, meta, bits, block_values):
    return {"identity": dict(meta), "storage_precision": f"int{bits}",
            "compute_precision": "f32", "block_values": block_values,
            "future_offsets": [head["future_offset"] for head in pack["heads"]]}


def export_onnx(pack, bits=4, block_values=64, expected_identity=None):
    """Return serialized opset-17/IR-8 ONNX with hidden[1,H] -> logits[K,V].

    Every head consumes the same hidden tensor through two MatMuls plus bias.
    Integer storage is decoded into F32 graph initializers, explicitly recorded
    in metadata. No model or execution-provider dependency is downloaded.
    """
    meta = heads.validate_pack(pack, expected_identity)
    onnx, np = _dependency("onnx"), _dependency("numpy")
    helper = onnx.helper
    initializers, nodes, rows = [], [], []
    for i, (head, (down, up)) in enumerate(zip(pack["heads"], _weights(pack, bits, block_values))):
        rank = head["rank"]
        name = f"head_{i}"
        tensors = {
            "down_t": np.asarray(down, dtype=np.float32).reshape(rank, meta["hidden_size"]).T.copy(),
            "up_t": np.asarray(up, dtype=np.float32).reshape(meta["vocab_size"], rank).T.copy(),
            "bias": np.asarray([precision.f32(v) for v in head["bias"]], dtype=np.float32),
        }
        for key, array in tensors.items():
            initializers.append(onnx.numpy_helper.from_array(array, name=f"{name}_{key}"))
        nodes.extend([
            helper.make_node("MatMul", ["hidden", f"{name}_down_t"], [f"{name}_latent"], name=f"{name}_down"),
            helper.make_node("MatMul", [f"{name}_latent", f"{name}_up_t"], [f"{name}_scores"], name=f"{name}_up"),
            helper.make_node("Add", [f"{name}_scores", f"{name}_bias"], [f"{name}_logits"], name=f"{name}_bias_add"),
        ])
        rows.append(f"{name}_logits")
    nodes.append(helper.make_node("Concat", rows, ["logits"], axis=0, name="head_concat"))
    graph = helper.make_graph(nodes, "colibri_strata_independent_heads",
                              [helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT, [1, meta["hidden_size"]])],
                              [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT,
                                                             [len(pack["heads"]), meta["vocab_size"]])],
                              initializer=initializers)
    model = helper.make_model(graph, producer_name="colibri-strata-experimental",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    properties = {key: str(value) for key, value in meta.items()}
    properties.update({"storage_precision": f"int{bits}", "compute_precision": "f32",
                       "block_values": str(block_values),
                       "future_offsets": json.dumps([head["future_offset"] for head in pack["heads"]]),
                       "identity": json.dumps(meta, sort_keys=True),
                       "weight_representation": "decoded-f32-initializers"})
    helper.set_model_props(model, properties)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _output(np, value, pack, meta):
    array = np.asarray(value)
    heads.require(array.shape == (len(pack["heads"]), meta["vocab_size"]), "runtime returned wrong logits shape")
    heads.require(array.dtype == np.dtype("float32"), "runtime returned non-F32 logits")
    heads.require(bool(np.all(np.isfinite(array))), "runtime returned nonfinite logits")
    return array.tolist(), [int(token) for token in np.argmax(array, axis=1)]


def _profile_providers(profile_path, directory, requested_provider):
    path, root = Path(profile_path).resolve(), Path(directory).resolve()
    if root not in path.parents or not path.is_file() or path.stat().st_size > MAX_PROFILE_BYTES:
        raise RuntimeError("ONNX profile is missing, outside its temporary directory, or too large")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_PROFILE_BYTES + 1)
        if len(payload) > MAX_PROFILE_BYTES:
            raise RuntimeError("ONNX node profile exceeds its size limit")
        events = json.loads(payload)
    except (ValueError, OSError) as exc:
        raise RuntimeError("ONNX node profile could not be read") from exc
    if not isinstance(events, list) or len(events) > MAX_PROFILE_EVENTS:
        raise RuntimeError("ONNX node profile has invalid or excessive events")
    providers = set()
    compute_nodes = 0
    for event in events:
        if not isinstance(event, dict):
            raise RuntimeError("ONNX node profile contains an invalid event")
        if event.get("cat") != "Node":
            continue
        name = event.get("name")
        if isinstance(name, str) and name.endswith(("_fence_before", "_fence_after")):
            continue
        if not isinstance(name, str) or not name.endswith("_kernel_time"):
            raise RuntimeError("ONNX node profile contains an unrecognized compute event")
        args = event.get("args")
        provider = args.get("provider") if isinstance(args, dict) else None
        if not isinstance(provider, str) or not provider:
            raise RuntimeError("ONNX compute node has no execution-provider proof")
        providers.add(provider)
        compute_nodes += 1
    if not compute_nodes or providers != {requested_provider}:
        raise RuntimeError(f"ONNX profile did not prove exclusive {requested_provider} compute; observed {sorted(providers)}")
    return sorted(providers), compute_nodes


def run_onnx(pack, hidden, provider="CPUExecutionProvider", provider_options=None,
             bits=4, block_values=64, expected_identity=None):
    """Run with one requested provider; reject missing providers/fallback.

    CPU is an explicit route. Accelerator requests disable CPU EP fallback;
    ORT rejects that configuration flag when CPU itself is explicitly selected.
    Python initialization/run fallback is disabled in every route. Provider
    options belong to the caller, including device-specific NPU configuration.
    Actual compute placement is verified from a temporary node profile.
    """
    heads.require(isinstance(provider, str) and provider and len(provider) <= 128, "invalid provider name")
    if provider_options is None:
        provider_options = {}
    heads.require(isinstance(provider_options, dict) and all(isinstance(k, str) and isinstance(v, str)
                  for k, v in provider_options.items()), "provider_options must map strings to strings")
    ort = _dependency("onnxruntime")
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"requested ONNX provider {provider!r} unavailable; available: {available}")
    np, meta, array = _input(pack, hidden, expected_identity)
    model = export_onnx(pack, bits, block_values, expected_identity)
    with tempfile.TemporaryDirectory(prefix="colibri-strata-ort-") as directory:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "nodes")
        if provider != "CPUExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if provider == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = None
        profile_finished = False
        try:
            session = ort.InferenceSession(model, sess_options=options,
                                           providers=[(provider, dict(provider_options))], enable_fallback=False)
            session.disable_fallback()
            if provider not in session.get_providers():
                raise RuntimeError("requested ONNX provider was substituted during initialization")
            output = session.run(["logits"], {"hidden": array})[0]
            profile_path = session.end_profiling()
            profile_finished = True
            executed, compute_nodes = _profile_providers(profile_path, directory, provider)
            logits, tokens = _output(np, output, pack, meta)
            result = _metadata(pack, meta, bits, block_values)
            result.update({"logits": logits, "tokens": tokens, "requested_provider": provider,
                           "executed_providers": executed, "profiled_compute_nodes": compute_nodes})
        finally:
            if session is not None and not profile_finished:
                try:
                    session.end_profiling()
                except Exception:
                    pass  # Preserve the original run/proof error; tempdir is removed.
            session = None
    return result


def run_jax(pack, hidden, platform="tpu", bits=4, block_values=64, expected_identity=None):
    """Explicit JAX cpu/gpu/tpu execution; unavailable platforms never substitute.

    Quantized storage is decoded into F32 tensors. Highest matmul precision is
    requested, inputs/weights are device_put on the selected device, the whole
    independent-head graph is jitted there, and completion/placement is checked.
    """
    heads.require(platform in ("cpu", "gpu", "tpu"), "JAX platform must be cpu, gpu, or tpu")
    jax = _dependency("jax")
    np, meta, array = _input(pack, hidden, expected_identity)
    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        raise RuntimeError(f"requested JAX platform {platform!r} unavailable") from exc
    if not devices or any(device.platform != platform for device in devices):
        raise RuntimeError(f"requested JAX platform {platform!r} unavailable; CPU substitution is forbidden")
    device = devices[0]
    jnp = _dependency("jax.numpy")
    tensors = []
    for head, (down, up) in zip(pack["heads"], _weights(pack, bits, block_values)):
        rank = head["rank"]
        tensors.append(tuple(jax.device_put(value, device) for value in (
            np.asarray(down, dtype=np.float32).reshape(rank, meta["hidden_size"]).T.copy(),
            np.asarray(up, dtype=np.float32).reshape(meta["vocab_size"], rank).T.copy(),
            np.asarray([precision.f32(v) for v in head["bias"]], dtype=np.float32))))

    def evaluate(h, weights):
        rows = [jnp.matmul(jnp.matmul(h, down, precision=jax.lax.Precision.HIGHEST), up,
                           precision=jax.lax.Precision.HIGHEST) + bias for down, up, bias in weights]
        return jnp.concatenate(rows, axis=0)

    placed = jax.device_put(array, device)
    placement = jax.sharding.SingleDeviceSharding(device)
    output = jax.jit(evaluate, in_shardings=placement, out_shardings=placement)(placed, tuple(tensors))
    output.block_until_ready()
    executed_devices = output.devices()
    if not executed_devices or any(d.platform != platform or d != device for d in executed_devices):
        raise RuntimeError("JAX output executed on an unexpected device/platform")
    logits, tokens = _output(np, np.asarray(output), pack, meta)
    result = _metadata(pack, meta, bits, block_values)
    result.update({"logits": logits, "tokens": tokens, "requested_platform": platform,
                   "executed_platform": platform, "executed_devices": sorted(str(d) for d in executed_devices)})
    return result

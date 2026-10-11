"""Optional real CPU execution plus strict accelerator placement/refusal tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import heads
import precision
import runtime


def available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


def pack():
    return {"schema": heads.PACK_SCHEMA,
            "identity": {"family": "qwen38", "checkpoint_revision": "a" * 40,
                         "tokenizer_sha256": "b" * 64, "hidden_tap": "synthetic_fixture_readout",
                         "quantization": "fixture-f32", "hidden_size": 3, "vocab_size": 5},
            "heads": [{"future_offset": 2, "rank": 2, "down": [1., 0., 0., 0., 1., 0.],
                       "up": [0., 0., 1., 0., 0., 1., 1., 1., -1., -1.], "bias": [0.] * 5},
                      {"future_offset": 5, "rank": 1, "down": [0., 0., 1.],
                       "up": [0., 1., -1., 2., 0.], "bias": [0., 0., 0., 0., 10.]}]}


class ProfileProofTests(unittest.TestCase):
    def check_profile(self, events, expected="CPUExecutionProvider", limit=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            if limit is None:
                return runtime._profile_providers(path, directory, expected)
            with mock.patch.object(runtime, "MAX_PROFILE_BYTES", limit):
                return runtime._profile_providers(path, directory, expected)

    @staticmethod
    def node(provider="CPUExecutionProvider"):
        return {"cat": "Node", "name": "head_0_up_kernel_time", "args": {"provider": provider}}

    def test_requires_actual_compute_node_proof(self):
        self.assertEqual(self.check_profile([self.node()]), (["CPUExecutionProvider"], 1))
        for events in ([], [{"cat": "Session", "args": {"provider": "CPUExecutionProvider"}}],
                       [{"cat": "Node", "name": "x_kernel_time", "args": {}}],
                       [self.node("QNNExecutionProvider")],
                       [self.node("QNNExecutionProvider"), self.node()],
                       [self.node(), {"cat": "Node", "name": "unknown_compute", "args": {"provider": "CPUExecutionProvider"}}],
                       {"provider": "CPUExecutionProvider"}):
            with self.subTest(events=events), self.assertRaises(RuntimeError):
                self.check_profile(events)

    def test_profile_must_be_bounded_and_local(self):
        with self.assertRaises(RuntimeError):
            self.check_profile([self.node()], limit=1)
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / "outside.json"
            path.write_text(json.dumps([self.node()]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runtime._profile_providers(path, directory, "CPUExecutionProvider")

    def test_missing_requested_provider_is_refused_before_session(self):
        fake = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        with mock.patch.object(runtime, "_dependency", return_value=fake):
            for provider in ("QNNExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider"):
                with self.subTest(provider=provider), self.assertRaisesRegex(RuntimeError, "unavailable"):
                    runtime.run_onnx(pack(), [1., 2., 3.], provider=provider)


@unittest.skipUnless(available("numpy", "onnx", "onnxruntime"), "optional numpy/onnx/onnxruntime unavailable")
class OnnxTests(unittest.TestCase):
    def test_export_static_shapes_independent_heads_and_metadata(self):
        import onnx
        import numpy as np
        p = pack()
        for bits in (4, 8, 16):
            model = onnx.load_from_string(runtime.export_onnx(p, bits, block_values=5))
            onnx.checker.check_model(model)
            self.assertEqual(model.ir_version, 8)
            self.assertEqual(model.opset_import[0].version, 17)
            self.assertEqual(model.graph.input[0].name, "hidden")
            self.assertEqual(model.graph.output[0].name, "logits")
            self.assertEqual([d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim], [1, 3])
            self.assertEqual([d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim], [2, 5])
            down_nodes = [node for node in model.graph.node if node.name.endswith("_down")]
            self.assertEqual(len(down_nodes), 2)
            self.assertTrue(all(node.input[0] == "hidden" for node in down_nodes))
            self.assertEqual([node.op_type for node in model.graph.node],
                             ["MatMul", "MatMul", "Add", "MatMul", "MatMul", "Add", "Concat"])
            meta = {row.key: row.value for row in model.metadata_props}
            self.assertEqual(meta["storage_precision"], f"int{bits}")
            self.assertEqual(meta["compute_precision"], "f32")
            self.assertEqual(json.loads(meta["future_offsets"]), [2, 5])
            self.assertEqual(json.loads(meta["identity"]), p["identity"])
            for tensor in model.graph.initializer:
                self.assertEqual(onnx.numpy_helper.to_array(tensor).dtype, np.dtype("float32"))

    def test_real_cpu_parity_all_precisions(self):
        import numpy as np
        p, hidden = pack(), [1., 2., 3.]
        for bits in (4, 8, 16):
            result = runtime.run_onnx(p, hidden, bits=bits, block_values=5)
            np.testing.assert_allclose(result["logits"], precision.reference(p, hidden, bits, 5), rtol=2e-5, atol=2e-5)
            self.assertEqual(result["tokens"], [3, 4])
            self.assertEqual(result["executed_providers"], ["CPUExecutionProvider"])
            self.assertGreater(result["profiled_compute_nodes"], 0)
            self.assertEqual(result["future_offsets"], [2, 5])
            self.assertEqual(result["compute_precision"], "f32")
            self.assertEqual(result["identity"], p["identity"])
        result = runtime.run_onnx(p, [[1., 2., 3.]], bits=8)
        self.assertEqual(result["tokens"], [3, 4])

    def test_bad_hidden_identity_and_bits(self):
        for hidden in ([1.], [[1., 2., 3.], [1., 2., 3.]], [float("nan"), 2., 3.],
                       [float("inf"), 2., 3.], [heads.FLOAT32_MAX * 2, 2., 3.],
                       [True, False, True], ["1", "2", "3"]):
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                runtime.run_onnx(pack(), hidden)
        changed = pack()["identity"] | {"tokenizer_sha256": "c" * 64}
        with self.assertRaises(ValueError):
            runtime.run_onnx(pack(), [1., 2., 3.], expected_identity=changed)
        for bits in (0, 3, 32, True):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                runtime.export_onnx(pack(), bits=bits)
        with self.assertRaises(ValueError):
            runtime.run_onnx(pack(), [1., 2., 3.], provider_options={"device_id": 0})

    def test_finite_output_validation(self):
        import numpy as np
        for output in (np.zeros((1, 5), dtype=np.float32), np.zeros((2, 5), dtype=np.float64),
                       np.full((2, 5), np.nan, dtype=np.float32), np.full((2, 5), np.inf, dtype=np.float32)):
            with self.assertRaises(ValueError):
                runtime._output(np, output, pack(), pack()["identity"])

    def fake_ort(self, events, requested="QNNExecutionProvider", initialized=None):
        import numpy as np
        captured = {}

        class Options:
            def __init__(self):
                self.config = {}

            def add_session_config_entry(self, key, value):
                self.config[key] = value

        class Session:
            def __init__(self, model, sess_options, providers, **kwargs):
                captured.update(options=sess_options, providers=providers, kwargs=kwargs,
                                directory=Path(sess_options.profile_file_prefix).parent)
                self.options = sess_options

            def disable_fallback(self):
                captured["fallback_disabled"] = True

            def get_providers(self):
                return initialized or [requested]

            def run(self, outputs, inputs):
                return [np.zeros((2, 5), dtype=np.float32)]

            def end_profiling(self):
                path = Path(self.options.profile_file_prefix + ".json")
                path.write_text(json.dumps(events), encoding="utf-8")
                return str(path)

        return SimpleNamespace(get_available_providers=lambda: [requested], SessionOptions=Options,
                               InferenceSession=Session, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential")), captured

    def test_ort_refuses_cpu_compute_and_initialization_substitution(self):
        real_dependency = runtime._dependency
        fake, captured = self.fake_ort([ProfileProofTests.node()])
        with mock.patch.object(runtime, "_dependency", side_effect=lambda name: fake if name == "onnxruntime" else real_dependency(name)):
            with self.assertRaisesRegex(RuntimeError, "did not prove"):
                runtime.run_onnx(pack(), [1., 2., 3.], provider="QNNExecutionProvider")
        self.assertEqual(captured["options"].config["session.disable_cpu_ep_fallback"], "1")
        self.assertFalse(captured["kwargs"]["enable_fallback"])
        self.assertTrue(captured["fallback_disabled"])
        self.assertEqual(captured["providers"], [("QNNExecutionProvider", {})])
        self.assertFalse(captured["directory"].exists())
        fake, captured = self.fake_ort([], initialized=["CPUExecutionProvider"])
        with mock.patch.object(runtime, "_dependency", side_effect=lambda name: fake if name == "onnxruntime" else real_dependency(name)):
            with self.assertRaisesRegex(RuntimeError, "substituted"):
                runtime.run_onnx(pack(), [1., 2., 3.], provider="QNNExecutionProvider")
        self.assertFalse(captured["directory"].exists())

    def test_directml_required_options_and_cleanup(self):
        fake, captured = self.fake_ort([ProfileProofTests.node("DmlExecutionProvider")], requested="DmlExecutionProvider")
        real_dependency = runtime._dependency
        with mock.patch.object(runtime, "_dependency", side_effect=lambda name: fake if name == "onnxruntime" else real_dependency(name)):
            result = runtime.run_onnx(pack(), [1., 2., 3.], provider="DmlExecutionProvider", provider_options={"device_id": "0"})
        self.assertFalse(captured["options"].enable_mem_pattern)
        self.assertEqual(captured["options"].execution_mode, "sequential")
        self.assertEqual(captured["providers"], [("DmlExecutionProvider", {"device_id": "0"})])
        self.assertEqual(result["executed_providers"], ["DmlExecutionProvider"])
        self.assertFalse(captured["directory"].exists())


@unittest.skipUnless(available("numpy", "jax"), "optional numpy/jax unavailable")
class JaxTests(unittest.TestCase):
    def test_explicit_cpu_parity_all_precisions(self):
        import numpy as np
        p, hidden = pack(), [1., 2., 3.]
        for bits in (4, 8, 16):
            result = runtime.run_jax(p, hidden, platform="cpu", bits=bits, block_values=5)
            np.testing.assert_allclose(result["logits"], precision.reference(p, hidden, bits, 5), rtol=2e-5, atol=2e-5)
            self.assertEqual(result["tokens"], [3, 4])
            self.assertEqual(result["executed_platform"], "cpu")
            self.assertTrue(result["executed_devices"])
            self.assertEqual(result["compute_precision"], "f32")

    def test_tpu_request_refuses_empty_or_cpu_devices(self):
        real_dependency = runtime._dependency
        for devices in ([], [SimpleNamespace(platform="cpu")]):
            fake = SimpleNamespace(devices=lambda platform: devices)
            with self.subTest(devices=devices), mock.patch.object(runtime, "_dependency", side_effect=lambda name: fake if name == "jax" else real_dependency(name)):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    runtime.run_jax(pack(), [1., 2., 3.], platform="tpu")
        fake = SimpleNamespace(devices=mock.Mock(side_effect=RuntimeError("TPU backend unavailable")))
        with mock.patch.object(runtime, "_dependency", side_effect=lambda name: fake if name == "jax" else real_dependency(name)):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                runtime.run_jax(pack(), [1., 2., 3.], platform="tpu")

    def test_jax_bad_input_and_identity(self):
        for hidden in ([1.], [float("nan"), 2., 3.]):
            with self.assertRaises(ValueError):
                runtime.run_jax(pack(), hidden, platform="cpu")
        with self.assertRaises(ValueError):
            runtime.run_jax(pack(), [1., 2., 3.], platform="cpu", expected_identity=pack()["identity"] | {"hidden_tap": "wrong"})
        with self.assertRaises(ValueError):
            runtime.run_jax(pack(), [1., 2., 3.], platform="npu")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Media boundaries and real-adapter contracts; model downloads are never tests."""
import base64
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stream_media as media
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experimental" / "strata"))
import reference_stream_models as models


def available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


WHISPER_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
                 "openvino_encoder_model.xml", "openvino_encoder_model.bin",
                 "openvino_decoder_model.xml", "openvino_decoder_model.bin",
                 "openvino_tokenizer.xml", "openvino_tokenizer.bin",
                 "openvino_detokenizer.xml", "openvino_detokenizer.bin")


def manifest(directory, speech=False):
    names = WHISPER_FILES if speech else ("model.xml", "model.bin")
    files = {}
    for name in names:
        payload = b"fixture-asset-" + name.encode()
        (Path(directory) / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    return {"schema": "colibri.runtime-model/v1", "family": "speech" if speech else "camera_upscale",
            "model_id": "unit-fixture", "adapter": "whisper_openvino" if speech else "super_resolution_openvino",
            "files": files, "precision": "fixture-f32", "max_input_bytes": 4 * 1024 * 1024,
            "resident_bytes": 1024, "workspace_bytes": 128 * 1024 * 1024, "session_bytes": 32 * 1024 * 1024,
            "options": {"model_path": "."} if speech else {"model_xml": "model.xml", "tile_overlap": 0}}


def wav_float(values):
    payload = b"".join(struct.pack("<f", v) for v in values)
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    return b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload


@unittest.skipUnless(available("numpy", "PIL"), "optional numpy/Pillow unavailable")
class MediaTests(unittest.TestCase):
    def test_pcm16_and_float_wav_boundaries(self):
        import numpy as np
        pcm = media.encode_wav_pcm16([-1., -0.5, 0., 0.5, 1.])
        decoded = media.decode_wav(pcm)
        self.assertEqual(decoded["sample_rate"], 16000)
        self.assertEqual(decoded["sample_count"], 5)
        np.testing.assert_allclose(decoded["samples"], [-1., -0.5, 0., 0.5, 32767 / 32768], atol=1e-7)
        floating = media.decode_wav(wav_float([-1., 0., 1.]))
        np.testing.assert_array_equal(floating["samples"], [-1., 0., 1.])
        self.assertEqual(media.validate_wav(pcm)["duration"], 5 / 16000)
        self.assertEqual(media.media_sha256(pcm), hashlib.sha256(pcm).hexdigest())

    def test_wav_rejects_size_format_nonfinite_and_long_input(self):
        base = bytearray(media.encode_wav_pcm16([0., 0.]))
        cases = [bytes(base[:-1]), bytes(base) + b"trailing", wav_float([float("nan")]),
                 wav_float([float("inf")]), wav_float([1.01]), wav_float([])]
        for offset, encoded in ((22, struct.pack("<H", 2)), (24, struct.pack("<I", 48000)),
                                (28, struct.pack("<I", 1)), (32, struct.pack("<H", 4)),
                                (34, struct.pack("<H", 8))):
            bad = bytearray(base); bad[offset:offset + len(encoded)] = encoded; cases.append(bytes(bad))
        for payload in cases:
            with self.subTest(payload=payload[:44]), self.assertRaises(ValueError):
                media.validate_wav(payload)
        with self.assertRaises(ValueError):
            media.validate_wav(bytes(base), max_bytes=1)
        with self.assertRaises(ValueError):
            media.encode_wav_pcm16([0.] * 480000)
        with self.assertRaises(ValueError):
            media.validate_wav(bytes(base), max_audio_seconds=float("nan"))
        self.assertEqual(media.validate_text("English 🦜")["text"], "English 🦜")
        for text in (b"\xff", "", "  ", "\ud800"):
            with self.assertRaises(ValueError): media.validate_text(text)

    def test_png_pixel_canary_alpha_and_bounds(self):
        import numpy as np
        rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [32, 64, 128]]], dtype=np.uint8)
        alpha = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        payload = media.encode_png(rgb, alpha)
        decoded = media.decode_png(payload)
        np.testing.assert_array_equal(decoded["rgb"], rgb)
        np.testing.assert_array_equal(decoded["alpha"], alpha)
        self.assertEqual(decoded["output_pixels"], 64)
        self.assertEqual(media.validate_png(payload)["channels"], 4)
        for bad in (payload[:-1], payload + b"x", b"notpng"):
            with self.assertRaises(ValueError):
                media.validate_png(bad)
        bad = bytearray(payload); bad[-5] ^= 1
        with self.assertRaises(ValueError):
            media.validate_png(bytes(bad))
        with self.assertRaises(ValueError):
            media.validate_png(payload, max_output_pixels=63)
        with self.assertRaises(ValueError):
            media.validate_png(payload, max_bytes=10)


@unittest.skipUnless(available("numpy", "PIL"), "optional numpy/Pillow unavailable")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.real_dependency = media.dependency

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def fake_ov(devices=None, execution=None, constant=None, fail=None, tile=(2, 3)):
        import numpy as np
        captured = {"tiles": []}
        devices = ["CPU", "GPU.0"] if devices is None else devices
        tile_h, tile_w = tile

        class Port:
            def __init__(self, shape): self.shape = shape

        class Model:
            inputs = [Port([1, 3, tile_h, tile_w]), Port([1, 3, tile_h * 4, tile_w * 4])]
            outputs = [Port([1, 3, tile_h * 4, tile_w * 4])]
            def input(self, name): return self.inputs[int(name)]
            def output(self, index): return self.outputs[index]

        class Request:
            def infer(self, inputs):
                captured["tiles"].append(inputs)
                if constant is not None:
                    out = np.full((1, 3, tile_h * 4, tile_w * 4), constant, dtype=np.float32)
                else:
                    out = np.repeat(np.repeat(inputs["0"] / 255, 4, axis=2), 4, axis=3)
                return {"output": out}

        class Compiled:
            def __init__(self, candidate): self.candidate = candidate
            def get_property(self, key):
                assert key == "EXECUTION_DEVICES"
                return execution if execution is not None else [self.candidate]
            def create_infer_request(self): return Request()
            def output(self, index): return "output"

        class Core:
            available_devices = devices
            def read_model(self, **kwargs): captured["assets"] = kwargs; return Model()
            def compile_model(self, model, device_name, config):
                captured.setdefault("compile_devices", []).append(device_name)
                if fail is not None and fail(device_name): raise RuntimeError("fixture compile rejection")
                return Compiled(device_name)

        return SimpleNamespace(Core=Core, Type=SimpleNamespace(f32="f32", f16="f16")), captured

    def dependencies(self, ov, genai=None):
        return mock.patch.object(media, "dependency", side_effect=lambda name: ov if name == "openvino" else genai if name == "openvino_genai" else self.real_dependency(name))

    def test_camera_tiles_restore_all_geometry_color_and_alpha(self):
        import numpy as np
        data = manifest(self.root)
        ov, captured = self.fake_ov(devices=["CPU"])
        rgb = np.zeros((5, 7, 3), dtype=np.uint8)
        rgb[:, :3] = [255, 0, 0]; rgb[:2, 3:] = [0, 255, 0]; rgb[2:, 3:] = [0, 0, 255]
        alpha = np.full((5, 7), 255, dtype=np.uint8)
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "CPU")
            result = adapter.upscale(media.encode_png(rgb, alpha))
        output = media.decode_png(base64.b64decode(result["png_base64"]), scale=1)
        np.testing.assert_array_equal(output["rgb"], np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1))
        self.assertEqual((result["width"], result["height"]), (28, 20))
        self.assertEqual(len(captured["tiles"]), 9)
        self.assertEqual(captured["tiles"][0]["0"][0, :, 0, 0].tolist(), [0., 0., 255.])
        self.assertEqual(captured["tiles"][0]["1"].shape, (1, 3, 8, 12))
        self.assertTrue(result["execution_devices_verified"])
        self.assertIsNotNone(output["alpha"])
        adapter.close()
        with self.assertRaises(RuntimeError): adapter.upscale(b"irrelevant")

    def test_overlap_and_thin_images_keep_coordinates_with_bounded_tile_padding(self):
        import numpy as np
        data = manifest(self.root); data["options"]["tile_overlap"] = 1
        ov, captured = self.fake_ov(devices=["CPU"], tile=(6, 8))
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "CPU")
            for height, width in ((7, 13), (1, 17), (11, 1)):
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                yy, xx = np.indices((height, width)); rgb[yy, xx, (yy + xx) % 3] = 255
                original_pad = np.pad
                with mock.patch.object(np, "pad", wraps=original_pad) as pad:
                    result = adapter.upscale(media.encode_png(rgb))
                output = media.decode_png(base64.b64decode(result["png_base64"]), scale=1)
                np.testing.assert_array_equal(output["rgb"], np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1))
                self.assertTrue(all(call.args[0].shape[0] <= 6 and call.args[0].shape[1] <= 8 for call in pad.call_args_list))

    def test_model_output_used_not_bicubic_and_numeric_failure(self):
        import numpy as np
        payload = media.encode_png(np.full((1, 1, 3), 255, dtype=np.uint8))
        data = manifest(self.root)
        ov, _ = self.fake_ov(devices=["CPU"], constant=0.25)
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "CPU")
            result = adapter.upscale(payload)
        output = media.decode_png(base64.b64decode(result["png_base64"]), scale=1)
        self.assertTrue(np.all(output["rgb"] == 63))
        ov, _ = self.fake_ov(devices=["CPU"], constant=float("nan"))
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "CPU")
            with self.assertRaises(ValueError): adapter.upscale(payload)

    def test_cancel_and_actual_device_refusal(self):
        import numpy as np
        data = manifest(self.root)
        payload = media.encode_png(np.zeros((5, 7, 3), dtype=np.uint8))
        ov, captured = self.fake_ov(devices=["CPU"])
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "CPU")
            with self.assertRaises(models.StreamCancelled): adapter.upscale(payload, cancel=lambda: True)
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                return calls >= 3
            with self.assertRaises(models.StreamCancelled): adapter.upscale(payload, cancel=cancel)
        self.assertLessEqual(len(captured["tiles"]), 1)
        ov, _ = self.fake_ov(devices=["GPU.0"], execution=["CPU"])
        with self.dependencies(ov), self.assertRaises(ValueError): models.make_adapter(data, self.root, "GPU")
        ov, _ = self.fake_ov(devices=["CPU"])
        with self.dependencies(ov), self.assertRaises(ValueError): models.make_adapter(data, self.root, "NPU")

    def test_auto_compilation_fallback_and_unsafe_assets(self):
        data = manifest(self.root)
        ov, captured = self.fake_ov(fail=lambda candidate: candidate.startswith("GPU"))
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "AUTO")
        self.assertEqual(captured["compile_devices"], ["GPU.0", "CPU"])
        self.assertEqual(adapter.info["device"], "CPU")
        self.assertEqual(len(adapter.info["auto_attempts"]), 1)
        data["options"]["model_xml"] = "../escape.xml"
        with self.dependencies(ov), self.assertRaises(ValueError): models.make_adapter(data, self.root)
        data["options"]["model_xml"] = "model.xml"
        (self.root / "model.bin").unlink()
        with self.dependencies(ov), self.assertRaises(ValueError): models.make_adapter(data, self.root)

    def test_whisper_english_translation_auto_language_and_cancellation(self):
        captured = {}
        ov, _ = self.fake_ov(devices=["CPU"])
        class Pipe:
            def __init__(self, path, device): captured["device"] = device
            def get_generation_config(self): return SimpleNamespace(language="<|en|>")
            def generate(self, samples, **kwargs):
                captured["samples"], captured["generation"] = samples, kwargs
                if captured.get("trigger_cancel"):
                    captured["cancelled"] = True
                captured["streamer_stop"] = kwargs["streamer"]("piece")
                return SimpleNamespace(texts=["English model text"])
        genai = SimpleNamespace(WhisperPipeline=Pipe)
        data = manifest(self.root, speech=True)
        with self.dependencies(ov, genai):
            adapter = models.make_adapter(data, self.root, "CPU")
            audio = media.encode_wav_pcm16([0., 0.5, -0.5])
            result = adapter.transcribe(audio)
            self.assertEqual(captured["generation"]["language"], "<|en|>")
            self.assertFalse(captured["generation"]["return_timestamps"])
            self.assertEqual(result["segments"], [])
            self.assertEqual(result["target_language"], "en")
            result = adapter.transcribe(audio, task="translate", language=None)
            self.assertEqual(captured["generation"]["task"], "translate")
            self.assertNotIn("language", captured["generation"])
            self.assertIsNone(captured["generation"]["generation_config"].language)
            self.assertEqual(result["target_language"], "en")
            captured["trigger_cancel"] = True
            with self.assertRaises(models.StreamCancelled):
                adapter.transcribe(audio, cancel=lambda: captured.get("cancelled", False))
            self.assertTrue(captured["streamer_stop"])
            captured.pop("trigger_cancel", None)
            polls = iter((False, False, True, False))
            with self.assertRaises(models.StreamCancelled):
                adapter.transcribe(audio, cancel=lambda: next(polls))
            with self.assertRaises(ValueError): adapter.transcribe(audio, language="fr")
            adapter.close()
            with self.assertRaises(RuntimeError): adapter.transcribe(audio)
        self.assertFalse(adapter.info["execution_devices_verified"])

    def test_affected_windows_speech_gpu_refused_auto_selects_cpu(self):
        ov, _ = self.fake_ov(devices=["GPU.0", "CPU"])
        constructed = []
        class Pipe:
            def __init__(self, path, device): constructed.append(device)
        genai = SimpleNamespace(WhisperPipeline=Pipe, __version__="2026.4.1.0-3408-test")
        data = manifest(self.root, speech=True)
        with self.dependencies(ov, genai), mock.patch.object(models.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "shutdown qualification"):
                models.make_adapter(data, self.root, "GPU")
            adapter = models.make_adapter(data, self.root, "AUTO")
        self.assertEqual(constructed, ["CPU"])
        self.assertEqual(adapter.info["device"], "CPU")
        self.assertEqual(adapter.info["auto_attempts"][0]["device"], "GPU.0")
        self.assertIn("reason", adapter.info["auto_attempts"][0])
        adapter.close(); adapter.close()

    def test_float_precision_hint_is_distinct_from_int16_storage(self):
        data = manifest(self.root)
        ov, _ = self.fake_ov(devices=["GPU"])
        data["options"]["inference_precision"] = "f16"
        with self.dependencies(ov):
            adapter = models.make_adapter(data, self.root, "GPU")
        self.assertEqual(adapter.info["inference_precision_hint"], "f16")
        self.assertEqual(adapter.info["runtime_precision"], "unknown")
        adapter.close()
        data["options"]["inference_precision"] = "int16"
        with self.dependencies(ov), self.assertRaises(ValueError): models.make_adapter(data, self.root, "GPU")

    def test_text_complete_chat_are_stateless_and_bounded(self):
        captured = {"prompts": []}
        ov, _ = self.fake_ov(devices=["CPU"])
        class Tokenizer:
            def encode(self, text):
                return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, len(text.split()) + 1)))
            def apply_chat_template(self, messages, add_generation_prompt):
                self.assert_generation = add_generation_prompt
                return " ".join(f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages) + " <assistant>"
        tokenizer = Tokenizer()
        class Pipe:
            def __init__(self, path, device): captured["device"] = device
            def get_tokenizer(self): return tokenizer
            def start_chat(self, *args): raise AssertionError("shared chat state must never start")
            def generate(self, prompt, **kwargs):
                captured["prompts"].append(prompt)
                captured["generation"] = kwargs
                kwargs["streamer"]("token")
                return "actual compact model output"
        genai = SimpleNamespace(LLMPipeline=Pipe)
        data = manifest(self.root)
        data.update(family="text_graph", adapter="text_openvino", max_input_bytes=1024)
        for name in ("config.json", "openvino_model.xml", "openvino_model.bin", "openvino_tokenizer.xml", "openvino_tokenizer.bin", "openvino_detokenizer.xml", "openvino_detokenizer.bin"):
            (self.root / name).write_bytes(b"fixture")
            data["files"][name] = hashlib.sha256(b"fixture").hexdigest()
        data["options"] = {"max_tokens": 256, "max_prompt_tokens": 32, "max_context_tokens": 256}
        with self.dependencies(ov, genai):
            adapter = models.make_adapter(data, self.root, "CPU")
            result = adapter.complete(b"English raw prompt", max_new_tokens=10)
            self.assertEqual(result["text"], "actual compact model output")
            self.assertEqual(result["prompt_token_count"], 4)
            adapter.chat([{"role": "user", "content": "first private question"}], max_new_tokens=10)
            adapter.chat([{"role": "user", "content": "second separate question"}], max_new_tokens=10)
            self.assertNotIn("first private", captured["prompts"][-1])
            self.assertTrue(tokenizer.assert_generation)
            self.assertFalse(adapter.info["stateful_chat"])
            for kwargs in ({"max_new_tokens": 257}, {"max_new_tokens": True}):
                with self.assertRaises(ValueError): adapter.complete("x", **kwargs)
            with self.assertRaises(ValueError): adapter.complete("word " * 40, max_new_tokens=1)
            with self.assertRaises(ValueError): adapter.complete(b"\xff")
            with self.assertRaises(ValueError): adapter.chat([{"role": "tool", "content": "unsupported"}])
            polls = iter((False, False, True, False))
            with self.assertRaises(models.StreamCancelled): adapter.complete("x", max_new_tokens=1, cancel=lambda: next(polls))
            adapter.close()
            with self.assertRaises(RuntimeError): adapter.complete("x")


@unittest.skipUnless(available("openvino", "numpy", "PIL"), "optional OpenVINO/numpy/Pillow unavailable")
class RealGraphFixtureTests(unittest.TestCase):
    def test_actual_cpu_ir_request(self):
        import numpy as np
        import openvino as ov
        op = ov.opset13
        with tempfile.TemporaryDirectory() as directory:
            low = op.parameter([1, 3, 2, 3], np.float32, name="0")
            high = op.parameter([1, 3, 8, 12], np.float32, name="1")
            low.output(0).get_tensor().set_names({"0"}); high.output(0).get_tensor().set_names({"1"})
            zero = op.multiply(op.reduce_sum(low, op.constant([0, 1, 2, 3], np.int64), True), op.constant(np.float32(0)))
            out = op.add(op.multiply(high, op.constant(np.float32(1 / 255))), zero)
            graph = ov.Model([out], [low, high], "fixture_not_a_trained_model")
            xml, binary = Path(directory) / "model.xml", Path(directory) / "model.bin"
            ov.serialize(graph, str(xml), str(binary))
            data = {"schema": "colibri.runtime-model/v1", "family": "camera_upscale", "model_id": "actual-IR-fixture",
                    "adapter": "super_resolution_openvino", "precision": "f32", "options": {"model_xml": "model.xml"},
                    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (xml, binary)}}
            adapter = models.make_adapter(data, directory, "CPU")
            result = adapter.upscale(media.encode_png(np.full((2, 3, 3), 128, dtype=np.uint8)))
            self.assertEqual((result["width"], result["height"]), (12, 8))
            self.assertEqual(result["device"], "CPU")
            adapter.close()


if __name__ == "__main__": unittest.main(verbosity=2)

"""Media export correctness and explicit-capture CLI contracts."""
import base64
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import runtime_cli

try:
    import av
    import numpy as np
    from PIL import Image
except ImportError:
    av = None


@unittest.skipIf(av is None, "optional PyAV/NumPy/Pillow media packages absent")
class VideoExport(unittest.TestCase):
    @staticmethod
    def fixture(path):
        with av.open(str(path), "w") as container:
            video = container.add_stream("libx264", rate=5)
            video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
            video.time_base = video.codec_context.time_base = Fraction(1, 1000)
            audio = container.add_stream("aac", rate=16000)
            audio.layout = "mono"
            audio.codec_context.time_base = Fraction(1, 16000)
            for index, stamp in enumerate((0, 200, 650)):
                pixels = np.zeros((48, 64, 3), dtype="uint8")
                pixels[:, :, index] = 160
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = stamp, Fraction(1, 1000)
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            for index in range(16):
                samples = np.zeros((1, 1024), dtype="float32")
                frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                frame.sample_rate = 16000
                frame.pts, frame.time_base = index * 1024, Fraction(1, 16000)
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)

    def test_preserves_all_frames_variable_timestamps_and_audio(self):
        class ExportAdapter:
            def upscale(self, data):
                with Image.open(io.BytesIO(data)) as image:
                    encoded = runtime_cli._png_bytes(image.resize((image.width * 4, image.height * 4)))
                return {"png_base64": base64.b64encode(encoded).decode("ascii")}
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "in.mp4", Path(directory) / "out.mp4"
            self.fixture(source)
            result = runtime_cli._video(ExportAdapter(), source, target)
            with av.open(str(source)) as reader:
                before = [float(f.pts * f.time_base) for f in reader.decode(video=0)]
            with av.open(str(target)) as reader:
                self.assertEqual(len(reader.streams.audio), 1)
                frames = list(reader.decode(video=0))
                after = [float(f.pts * f.time_base) for f in frames]
                self.assertEqual([(f.width, f.height) for f in frames], [(256, 192)] * 3)
            with av.open(str(target)) as reader:
                samples = sum(f.samples for f in reader.decode(audio=0))
                self.assertGreater(samples, 0)
            self.assertEqual(result["frames"], 3)
            self.assertEqual(result["audio_streams"], 1)
            np.testing.assert_allclose(after, before, atol=1 / 90000)
            with self.assertRaises(FileExistsError):
                runtime_cli._video(ExportAdapter(), source, target)

    def test_image_reader_bounds_and_output_preserves_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            Image.new("RGB", (64, 48), (10, 20, 30)).save(path)
            data = runtime_cli._image(path)
            with Image.open(io.BytesIO(data)) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (64, 48))
            with self.assertRaises(FileExistsError):
                runtime_cli._output(path, b"replacement")
            with self.assertRaises(ValueError):
                runtime_cli._file(path, 1)

    def test_palette_partial_transparency_survives_cli_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "palette.png"
            image = Image.new("P", (2, 1))
            image.putpalette([255, 0, 0, 0, 255, 0] + [0] * (768 - 6))
            image.putdata([0, 1])
            image.save(path, transparency=bytes([128, 255]))
            with Image.open(io.BytesIO(runtime_cli._image(path))) as result:
                self.assertEqual(list(result.getchannel("A").tobytes()), [128, 255])


class CliContracts(unittest.TestCase):
    def test_budget_and_native_flags_are_enforced(self):
        self.assertEqual(runtime_cli.budget(SimpleNamespace(ram=2)), 2 << 30)
        for options in ({"vram": 8}, {"xdna": True}, {"cluster_coordinator": "http://invalid"},
                        {"temp": 0}, {"memory_budget_mib": -1}, {"ram": -1},
                        {"ram": 2, "memory_budget_mib": 2048}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runtime_cli.budget(SimpleNamespace(**options))

    def test_direct_server_script_uses_its_own_http_error_class(self):
        import runpy
        namespace = runpy.run_path(str(HERE / "openai_server.py"), run_name="script_import_check")
        handler = namespace["APIHandler"].__new__(namespace["APIHandler"])
        error = handler.runtime_error(400, "invalid")
        self.assertIsInstance(error, namespace["APIError"])

    def test_help_exposes_graph_and_explicit_capture(self):
        for command, expected in (("serve", "--device"), ("transcribe", "--microphone"),
                                  ("upscale", "--camera"), ("prepare-runtime", "text_graph")):
            result = subprocess.run([sys.executable, str(HERE / "coli"), command, "--help"],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()

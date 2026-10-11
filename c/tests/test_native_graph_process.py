"""SDK-free native PNG admission regressions; no model/device is loaded.

Build a private C executable, then exercise the real --validate route. The
deflate padding fixture is deliberately only8MiB to prove the bound safely.
"""
from pathlib import Path
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
PASSES = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
          (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def pixel(x, y, color):
    rgb = ((x * 37 + y * 19 + 9) & 255,
           (x * 11 + y * 7 + 39) & 255,
           (x * 3 + y * 31 + 73) & 255)
    alpha = (x * 17 + y * 23 + 91) & 255
    return bytes({0: (rgb[0],), 2: rgb, 3: ((x + y) % 3,),
                  4: (rgb[0], alpha), 6: (*rgb, alpha)}[color])


def scanlines(width, height, color, interlace):
    data = bytearray()
    passes = PASSES if interlace else ((0, 0, 1, 1),)
    for x0, y0, dx, dy in passes:
        if width <= x0 or height <= y0:
            continue
        for y in range(y0, height, dy):
            data.append(0)  # PNG filter0; every pixel is already a source U8.
            for x in range(x0, width, dx):
                data.extend(pixel(x, y, color))
    return bytes(data)


def png(width=3, height=2, color=2, interlace=0, *, raw=None,
        split_idat=False, depth=8):
    if raw is None:
        raw = scanlines(width, height, color, interlace)
    header = struct.pack(">2I5B", width, height, depth, color, 0, 0, interlace)
    result = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    if color == 3:
        result += chunk(b"PLTE", bytes((0, 0, 0, 255, 80, 0, 9, 30, 255)))
        result += chunk(b"tRNS", bytes((0, 128, 255)))
    compressed = zlib.compress(raw)
    if split_idat:
        midpoint = len(compressed) // 2
        result += chunk(b"IDAT", compressed[:midpoint])
        result += chunk(b"IDAT", compressed[midpoint:])
    else:
        result += chunk(b"IDAT", compressed)
    return result + chunk(b"IEND", b"")


class NativePNGProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="colibri-png-\u00e9-\u65e5\u672c-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.binary = Path(cls.temporary.name) / ("native_graph.exe" if os.name == "nt" else "native_graph")
        command = [sys.executable, str(ROOT / "tools/build_native_graph.py"),
                   "--output", str(cls.binary)]
        if os.environ.get("COLI_NATIVE_GRAPH_TEST_SANITIZE") == "1":
            command.append("--sanitize")
        built = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
        if built.returncode:
            raise AssertionError("private native validator build failed:\n" +
                                 built.stdout[-4000:] + built.stderr[-4000:])

    def validate(self, payload, *, max_pixels=5592405, max_bytes=4 << 20):
        return subprocess.run([str(self.binary), "--validate", "camera",
                               "--max-pixels", str(max_pixels),
                               "--max-bytes", str(max_bytes)], input=payload,
                              capture_output=True, timeout=5)

    def reject(self, payload, **limits):
        result = self.validate(payload, **limits)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"", "invalid input published metadata")

    def test_noninterlaced_supported_colors_and_alpha(self):
        for color in CHANNELS:
            with self.subTest(color=color):
                result = self.validate(png(5, 7, color))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout),
                                 {"width": 5, "height": 7, "output_pixels": 560})

    def test_adam7_small_and_asymmetric_passes(self):
        for width, height in ((1, 1), (1, 9), (9, 1), (2, 3), (9, 6), (17, 13)):
            for color in CHANNELS:
                with self.subTest(width=width, height=height, color=color):
                    result = self.validate(png(width, height, color, 1, split_idat=True))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)["output_pixels"], width * height * 16)

    def test_exact_pixel_and_input_byte_limits(self):
        valid = png(3, 2)
        result = self.validate(valid, max_pixels=96, max_bytes=len(valid))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.reject(valid, max_pixels=95)
        self.reject(valid, max_bytes=len(valid) - 1)

    def test_crc_truncation_and_trailing_file_refused(self):
        valid = png()
        bad_crc = bytearray(valid)
        bad_crc[29] ^= 1  # IHDR CRC, while geometry and IDAT remain valid.
        self.reject(bytes(bad_crc))
        self.reject(valid[:-1])
        self.reject(valid + b"trailing")
        self.reject(png(0, 2))
        self.reject(png(3, 2, depth=16))

    def test_decoded_padding_and_short_scanline_refused(self):
        for interlace in (0, 1):
            raw = scanlines(9, 6, 6, interlace)
            with self.subTest(interlace=interlace):
                self.reject(png(9, 6, 6, interlace, raw=raw + b"\0"))
                self.reject(png(9, 6, 6, interlace, raw=raw[:-1]))

    def test_tiny_ihdr_cannot_hide_massive_inflate(self):
        payload = png(1, 1, 2, raw=b"\0" * (8 << 20))
        self.assertLess(len(payload), 16384)
        self.reject(payload, max_pixels=16, max_bytes=65536)

    def test_palette_indices_outside_declared_entries_refused(self):
        # Valid framing/CRC and exact inflate length must not allow an index
        # into uninitialized entries in the decoder's stack palette.
        self.reject(png(1, 1, 3, raw=b"\0\xff"))
        self.reject(png(1, 1, 3, 1, raw=b"\0\xff"))


if __name__ == "__main__":
    unittest.main()

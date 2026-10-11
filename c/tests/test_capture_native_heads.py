"""Native future labels never cross a request boundary or skip a position."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from capture_native_heads import PRECISION, convert_raw


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name) / "raw.jsonl"
        self.output = Path(self.tmp.name) / "training.jsonl"
        self.expected = {"fingerprint": "a" * 64, "hidden_size": 4,
                         "vocab_size": 32, "tokenizer_sha256": "b" * 64}
        self.header = {"schema": "colibri.native-hidden-raw/v1",
                       "native_fingerprint": "a" * 64, "hidden_size": 4,
                       "vocab_size": 32, "quantization": PRECISION,
                       "hidden_tap": "final-rmsnorm+next-embedding/v1"}

    def rows(self, session, tokens):
        return [{"session": session, "position": index + 10,
                 "hidden": [0.5, -0.5, 0.0, 1.0], "next_token": token}
                for index, token in enumerate(tokens)]

    def write(self, rows):
        self.raw.write_text("\n".join(json.dumps(r) for r in [self.header, *rows]) + "\n")

    def convert(self):
        return convert_raw(self.raw, self.output, "c" * 40, self.expected)

    def test_future_labels_and_sessions(self):
        self.write(self.rows(1, [1, 2, 3]) + self.rows(2, [7, 8, 9]))
        self.assertEqual(self.convert(), 4)
        header, *rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(header["native_fingerprint"], "a" * 64)
        self.assertEqual([r["future_tokens"] for r in rows], [[1, 2, 3], [2, 3], [7, 8, 9], [8, 9]])
        self.assertEqual([r["session"] for r in rows], [1, 1, 2, 2])

    def test_discontinuity_refused_before_output(self):
        rows = self.rows(1, [1, 2, 3]); rows[1]["position"] += 1
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_changed_identity_refused(self):
        self.header["native_fingerprint"] = "d" * 64
        self.write(self.rows(1, [1, 2]))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_bad_numeric_feature_or_label_refused(self):
        for replacement in (float("nan"), float("inf"), 1e100, True):
            with self.subTest(feature=replacement):
                rows = self.rows(1, [1, 2]); rows[0]["hidden"][0] = replacement
                self.write(rows)
                with self.assertRaisesRegex(ValueError, "feature/label"):
                    self.convert()
                self.assertFalse(self.output.exists())
        self.write(self.rows(1, [32, 2]))
        with self.assertRaisesRegex(ValueError, "feature/label"):
            self.convert()

    def test_existing_training_file_preserved(self):
        self.write(self.rows(1, [1, 2]))
        self.output.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(self.output.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()

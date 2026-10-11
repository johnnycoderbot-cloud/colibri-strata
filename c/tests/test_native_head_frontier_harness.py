"""Synthetic fixture mode and preflight tests; no native engine execution."""
import copy
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("native_head_frontier_harness", Path(__file__).with_name("native_head_frontier_harness.py"))
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.args = SimpleNamespace(model=self.model, work=self.root, revision="0" * 40, bank_kind="ordinary")
        self.native = {"hidden_size": 4, "vocab_size": 9, "tokenizer_sha256": "a" * 64, "fingerprint": "b" * 64}
        self.captured = []

    def export(self, pack, model, precision, *, bits, kind):
        self.captured.append((copy.deepcopy(pack), kind))
        version, reserved = (2, 1) if kind == "logit-residual" else (1, 0)
        return struct.pack("<4s7I", b"CSNH", version, 96, 4, 9, len(pack["heads"]), bits, reserved) + bytes(64)

    def test_ordinary_fixture_biases_and_v1_format_are_unchanged(self):
        with patch.object(harness.exporter, "export_pack", self.export):
            path = harness.constant_bank(self.args, self.native, "ordinary", [2, 4])
        pack, kind = self.captured[0]
        self.assertEqual(kind, "ordinary")
        self.assertEqual(pack["schema"], harness.exporter.heads.PACK_SCHEMA)
        self.assertNotIn("kind", pack)
        self.assertTrue(pack["training"]["synthetic_correctness_fixture_only"])
        self.assertFalse(pack["training"]["quality_validated"])
        self.assertEqual([head["future_offset"] for head in pack["heads"]], [2, 3])
        self.assertEqual(pack["heads"][0]["bias"], [0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual((struct.unpack_from("<4s7I", path.read_bytes())[1],
                          struct.unpack_from("<4s7I", path.read_bytes())[7]), (1, 0))

    def test_residual_finite_biases_dominate_base_and_require_v2_format(self):
        self.args.bank_kind = "logit-residual"
        with patch.object(harness.exporter, "export_pack", self.export):
            path = harness.constant_bank(self.args, self.native, "residual", [2, 4])
        pack, kind = self.captured[0]
        self.assertEqual(kind, "logit-residual")
        self.assertEqual(pack["schema"], "colibri.strata.logit-residual-heads/v1")
        self.assertEqual(pack["kind"], "logit-residual")
        self.assertTrue(pack["training"]["synthetic_correctness_fixture_only"])
        for index, token in enumerate((2, 4)):
            head = pack["heads"][index]
            self.assertTrue(all(math.isfinite(value) for value in head["bias"]))
            self.assertEqual(head["bias"][token], 1e6)
            self.assertTrue(all(value == -1e6 for n, value in enumerate(head["bias"]) if n != token))
            self.assertEqual(head["down"], [0] * 4)
            self.assertEqual(head["up"], [0] * 9)
        header = struct.unpack_from("<4s7I", path.read_bytes())
        self.assertEqual((header[1], header[7]), (2, 1))

    def test_exporter_mode_mismatch_and_invalid_tokens_publish_no_fixture(self):
        self.args.bank_kind = "logit-residual"
        ordinary = struct.pack("<4s7I", b"CSNH", 1, 96, 4, 9, 1, 4, 0) + bytes(64)
        with patch.object(harness.exporter, "export_pack", return_value=ordinary):
            with self.assertRaisesRegex(AssertionError, "bank kind"):
                harness.constant_bank(self.args, self.native, "wrong-kind", [2])
        self.assertFalse((self.root / "wrong-kind.snh").exists())
        with patch.object(harness.exporter, "export_pack") as export:
            for tokens in ([], [-1], [9], [True], [2] * 9):
                with self.subTest(tokens=tokens), self.assertRaisesRegex(AssertionError, "valid vocabulary"):
                    harness.constant_bank(self.args, self.native, "invalid", tokens)
            export.assert_not_called()

    def test_existing_fixture_is_preserved(self):
        path = self.root / "existing.snh"
        path.write_bytes(b"preserved fixture")
        with patch.object(harness.exporter, "export_pack", self.export), self.assertRaises(FileExistsError):
            harness.constant_bank(self.args, self.native, "existing", [2])
        self.assertEqual(path.read_bytes(), b"preserved fixture")


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.engine = self.root / "native"
        self.engine.write_bytes(b"no execution fixture")

    def argv(self, *extra):
        return ["--engine", str(self.engine), "--model", str(self.model),
                "--work", str(self.root / "new-work"), "--output", str(self.root / "report.json"), *extra]

    def test_defaults_and_explicit_residual_selection_create_nothing(self):
        args = harness.parse_args(self.argv())
        self.assertEqual(args.bank_kind, "ordinary")
        self.assertEqual(args.selected_cases, set(harness.CASE_NAMES))
        args = harness.parse_args(self.argv("--bank-kind", "logit-residual", "--cases", "first-reject,cancel-mid-verify"))
        self.assertEqual(args.bank_kind, "logit-residual")
        self.assertEqual(args.selected_cases, {"first-reject", "cancel-mid-verify"})
        self.assertFalse(args.work.exists())
        self.assertFalse(args.output.exists())

    def test_invalid_cases_fail_before_engine_start_or_artifact_creation(self):
        with patch.object(harness, "Driver") as driver:
            for cases in ("", "first-reject,first-reject", "bogus", "first-reject, bogus"):
                with self.subTest(cases=cases), self.assertRaisesRegex(AssertionError, "case selection"):
                    harness.main(self.argv("--cases", cases))
            driver.assert_not_called()
        self.assertFalse((self.root / "new-work").exists())
        self.assertFalse((self.root / "report.json").exists())

    def test_existing_work_or_report_and_model_asset_paths_are_refused(self):
        (self.root / "report.json").write_bytes(b"preserved report")
        with self.assertRaisesRegex(AssertionError, "must be new"):
            harness.parse_args(self.argv())
        self.assertEqual((self.root / "report.json").read_bytes(), b"preserved report")
        with self.assertRaisesRegex(AssertionError, "outside model"):
            harness.parse_args(self.argv("--output", str(self.root / "other.json"),
                                          "--work", str(self.model / "new-work")))
        self.assertFalse((self.model / "new-work").exists())

    def test_plain_driver_clears_ambient_bank_and_trace(self):
        args = SimpleNamespace(engine=self.engine, model=self.model, work=self.root, threads=8, wsl=None)
        with patch.object(harness, "NativeEngine") as engine:
            harness.Driver(args, "plain", capture=False)
        env = engine.call_args.args[3]
        self.assertEqual(env["COLI_STRATA_HEADS"], "")
        self.assertEqual(env["COLI_STRATA_TRACE"], "")
        self.assertEqual(env["COLI_LOOKUP_FORCE"], "")


if __name__ == "__main__":
    unittest.main()

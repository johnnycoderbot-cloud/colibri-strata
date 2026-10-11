from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import aipu_check as aipu_check
import sys


class AipuCheckTests(unittest.TestCase):
    def test_existing_companion_diagnostic_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.json"; model.write_text("{}")
            report = root / "report.json"
            log = report.with_suffix(".stderr.log"); log.write_text("keep")
            argv = ["check", "--sdk-python", str(root), "--runtime-bin", str(root),
                    "--device-dir", str(root), "--sdk-entry", "fixture.runtime:entrypoint_main",
                    "--device-env", "FIXTURE_DEVICE_DIR", "--model", str(model), "--out", str(report)]
            with patch.object(sys, "argv", argv), patch.object(aipu_check, "invoke") as invoke:
                with self.assertRaises(SystemExit) as failure:
                    aipu_check.main()
            self.assertEqual(failure.exception.code, 2)
            invoke.assert_not_called()
            self.assertEqual(log.read_text(), "keep")
            self.assertFalse(report.exists())

    def test_every_frame_and_partial_summary(self):
        self.assertEqual(aipu_check.verification_counts("validation: ok validated all 123 checked frames", 0), (True, 123, 123))
        self.assertEqual(aipu_check.verification_counts("validation: ok validated 100.0% of 123 frames (checked=123 not_validated=0)", 0), (True, 123, 123))
        self.assertFalse(aipu_check.verification_counts("validation: ok validated 99.6% of 123 frames (checked=122 not_validated=1)", 0)[0])

    def test_no_fabricated_success(self):
        for text, code in (("", 0), ("validation: ok validated all 0 checked frames", 0),
                           ("validation: ok validated all 5 checked frames", 1),
                           ("validation: ok validated all 5 checked frames\nvalidation: FAILED", 0),
                           ("validation: ok validated all 5 checked frames\nvalidation: ok validated all 2 checked frames", 0)):
            self.assertFalse(aipu_check.verification_counts(text, code)[0])

    def test_explicit_paths_utf8_exit_propagation_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            python = root / "python.exe"; python.write_bytes(b"fixture")
            runtime = root / "bin"; runtime.mkdir()
            device = root / "device"; device.mkdir()
            completed = subprocess.CompletedProcess([], 1, "", "ERROR: fixture")
            with patch.object(aipu_check.subprocess, "run", return_value=completed) as run:
                result = aipu_check.invoke(python, runtime, device, ["--help"], sdk_entry="fixture.runtime:entrypoint_main", device_env="FIXTURE_DEVICE_DIR")
            self.assertEqual(result.returncode, 1)
            argv = run.call_args.args[0]
            self.assertEqual(argv[:3], [str(python), "-X", "utf8"])
            self.assertIn("sys.exit(getattr(importlib.import_module(module),function)())", argv[4])
            self.assertEqual(argv[5], "fixture.runtime:entrypoint_main")
            self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            with patch.object(aipu_check.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 30)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    aipu_check.invoke(python, runtime, device, [], sdk_entry="fixture.runtime:entrypoint_main", device_env="FIXTURE_DEVICE_DIR")
            with self.assertRaises(ValueError):
                aipu_check.invoke(Path("relative.exe"), runtime, device, [], sdk_entry="fixture.runtime:entrypoint_main", device_env="FIXTURE_DEVICE_DIR")


if __name__ == "__main__":
    unittest.main()

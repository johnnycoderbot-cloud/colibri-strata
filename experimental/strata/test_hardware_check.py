from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import heads


class HardwareCliTests(unittest.TestCase):
    def test_missing_pack_cannot_report_accelerator_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flag, value in (("--jax-platform", "tpu"), ("--provider", "MissingNPUExecutionProvider")):
                report = root / "report.json"
                result = subprocess.run([sys.executable, str(heads.ROOT / "experimental/strata/hardware_check.py"),
                                         flag, value, "--out", str(report)], capture_output=True,
                                        text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertIn("require --pack", result.stderr)
                self.assertFalse(report.exists())
                self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

"""SDK-free launcher tests; native SDK/model replays are a separate gate."""
import argparse
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mojo_cli


class MojoCLI(unittest.TestCase):
    def args(self, path, **overrides):
        values = dict(native=str(path), timeout=1, mojo_cmd="run", model="model.mef",
                      inputs="inputs.json", output="new-output", device="CPU", repeat=1)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_native_request_is_shell_free_and_has_exact_device(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "mojo_runtime"
            binary.write_bytes(b"fixture")
            with patch.object(mojo_cli.sys, "platform", "linux"), patch.object(mojo_cli.subprocess, "run") as run:
                run.return_value.returncode = 0
                self.assertEqual(mojo_cli.run(self.args(binary, device="ACCELERATOR.3", repeat=9)), 0)
                command = run.call_args.args[0]
                self.assertEqual(command[0], str(binary.resolve()))
                self.assertEqual(command[-4:], ["--device", "ACCELERATOR.3", "--repeat", "9"])
                self.assertEqual(run.call_args.kwargs, {"timeout": 1, "check": False})

    def test_invalid_devices_budgets_and_windows_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "mojo_runtime"
            binary.touch()
            with patch.object(mojo_cli.sys, "platform", "linux"), patch.object(mojo_cli.subprocess, "run") as run:
                for changes in ({"device": "NPU"}, {"device": "TPU"}, {"device": "AIPU"},
                                {"device": "AUTO"}, {"device": "ACCELERATOR.-1"},
                                {"timeout": float("nan")}, {"timeout": 0}, {"repeat": 10001}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        mojo_cli.run(self.args(binary, **changes))
                run.assert_not_called()
            with patch.object(mojo_cli.sys, "platform", "win32"):
                with self.assertRaisesRegex(RuntimeError, "WSL"):
                    mojo_cli.run(self.args(binary))

    def test_timeout_and_native_failure_are_not_cpu_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "mojo_runtime"
            binary.touch()
            with patch.object(mojo_cli.sys, "platform", "linux"), patch.object(mojo_cli.subprocess, "run") as run:
                run.return_value.returncode = 2
                self.assertEqual(mojo_cli.run(self.args(binary, mojo_cmd="probe")), 2)
                self.assertEqual(run.call_args.args[0], [str(binary.resolve()), "--probe"])
                run.side_effect = subprocess.TimeoutExpired("native", 1)
                with self.assertRaises(subprocess.TimeoutExpired):
                    mojo_cli.run(self.args(binary))
                self.assertEqual(run.call_count, 2)

    def test_launcher_imports_no_python_sdk(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"max", "mojo", "numpy", "torch", "openvino"}:
                raise AssertionError(name)
            return original(name, *args, **kwargs)
        with patch.object(builtins, "__import__", guarded):
            importlib.reload(mojo_cli)


if __name__ == "__main__":
    unittest.main()

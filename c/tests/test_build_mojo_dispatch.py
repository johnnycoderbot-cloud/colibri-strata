"""SDK-free build admission/atomic publication tests; C/Mojo replays are separate."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_mojo_dispatch


class MojoDispatchBuild(unittest.TestCase):
    def paths(self, temp):
        root = Path(temp)
        sdk = root / "SDK with spaces"
        (sdk / "lib/mojo").mkdir(parents=True)
        (sdk / "lib/mojo/std.mojoc").write_bytes(b"admission fixture")
        compiler = root / "compiler with spaces"
        compiler.write_bytes(b"admission fixture")
        output = root / "output with spaces" / "libdispatch.so"
        args = ["--compiler", str(compiler), "--sdk-root", str(sdk),
                "--output", str(output)]
        return sdk, compiler, output, args

    def test_actual_compiler_receives_shell_free_owned_sdk_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk, compiler, output, args = self.paths(temp)
            def compile_candidate(command, **kw):
                self.assertEqual(command[0], str(compiler.resolve()))
                self.assertEqual(command[1], "build")
                self.assertIn("shared-lib", command)
                self.assertIn("contract=off", command)
                self.assertEqual(kw["check"], True)
                self.assertNotIn("shell", kw)
                self.assertEqual(kw["env"]["MODULAR_HOME"], str(output.resolve()) + ".sdk")
                Path(command[-1]).write_bytes(b"new compiled native library")
            with patch.object(build_mojo_dispatch.sys, "platform", "linux"), \
                 patch.object(build_mojo_dispatch.subprocess, "run", side_effect=compile_candidate) as run, \
                 patch.dict(build_mojo_dispatch.os.environ, {"MODULAR_HOME": "unrelated SDK"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(build_mojo_dispatch.main(args), 0)
                self.assertEqual(build_mojo_dispatch.os.environ["MODULAR_HOME"], "unrelated SDK")
                self.assertEqual(run.call_count, 1)
            self.assertEqual(output.read_bytes(), b"new compiled native library")
            self.assertIn(str(sdk.resolve()), Path(str(output) + ".sdk/modular.cfg").read_text(encoding="utf-8"))
            self.assertEqual(list(output.parent.glob(".mojo-dispatch-build-*")), [])

    def test_compile_failure_preserves_previous_library(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, output, args = self.paths(temp)
            output.parent.mkdir()
            output.write_bytes(b"previous qualified native library")
            with patch.object(build_mojo_dispatch.sys, "platform", "linux"), \
                 patch.object(build_mojo_dispatch.subprocess, "run",
                              side_effect=subprocess.CalledProcessError(1, "compiler")):
                with self.assertRaises(subprocess.CalledProcessError):
                    build_mojo_dispatch.main(args)
            self.assertEqual(output.read_bytes(), b"previous qualified native library")
            self.assertEqual(list(output.parent.glob(".mojo-dispatch-build-*")), [])

    def test_windows_resources_and_conflicting_configuration_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk, _, output, args = self.paths(temp)
            config = Path(str(output) + ".sdk")
            config.mkdir(parents=True)
            (config / "modular.cfg").write_text("user owned config\n", encoding="utf-8")
            with patch.object(build_mojo_dispatch.subprocess, "run") as run, \
                 contextlib.redirect_stderr(io.StringIO()):
                with patch.object(build_mojo_dispatch.sys, "platform", "win32"), self.assertRaises(SystemExit):
                    build_mojo_dispatch.main(args)
                with patch.object(build_mojo_dispatch.sys, "platform", "linux"), self.assertRaises(SystemExit):
                    build_mojo_dispatch.main(args)
                (sdk / "lib/mojo/std.mojoc").unlink()
                with patch.object(build_mojo_dispatch.sys, "platform", "linux"), self.assertRaises(SystemExit):
                    build_mojo_dispatch.main(args)
                run.assert_not_called()
            self.assertEqual((config / "modular.cfg").read_text(encoding="utf-8"), "user owned config\n")

    def test_tool_imports_no_python_inference_sdk(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"max", "mojo", "numpy", "torch", "openvino"}:
                raise AssertionError(name)
            return original(name, *args, **kwargs)
        with patch.object(builtins, "__import__", guarded):
            importlib.reload(build_mojo_dispatch)


if __name__ == "__main__":
    unittest.main()

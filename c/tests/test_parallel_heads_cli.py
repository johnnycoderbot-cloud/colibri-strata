import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ParallelLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = importlib.machinery.SourceFileLoader("coli_parallel_test", str(Path(__file__).resolve().parents[1] / "coli"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.cli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.cli)

    def test_qualified_bank_resolves_and_preserves_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heads.snh"; path.write_bytes(b"CSNH")
            a = types.SimpleNamespace(parallel_heads=str(path), head_input="next-embedding")
            result = self.cli.parallel_heads_env(a, "qwen36", {"OMP_NUM_THREADS": "8"})
            self.assertEqual(result["COLI_STRATA_HEADS"], str(path.resolve()))
            self.assertEqual(result["COLI_STRATA_INPUT"], "next-embedding")
            self.assertEqual(result["OMP_NUM_THREADS"], "8")

    def test_other_family_refuses_inherited_pack_or_trace(self):
        a = types.SimpleNamespace(parallel_heads=None, head_input=None)
        for arch in ("glm", "olmoe", "qwen38", "camera_upscale", "qwen_image"):
            for env in ({"COLI_STRATA_HEADS": "pack.snh"}, {"COLI_STRATA_TRACE": "trace.jsonl"}):
                with self.subTest(arch=arch, env=env), self.assertRaisesRegex(ValueError, "other families are pending"):
                    self.cli.parallel_heads_env(a, arch, env)

    def test_missing_pack_and_input_without_pack_refused(self):
        with self.assertRaisesRegex(ValueError, "existing CSNH"):
            self.cli.parallel_heads_env(types.SimpleNamespace(parallel_heads="missing.snh", head_input=None), "qwen36", {})
        with self.assertRaisesRegex(ValueError, "requires"):
            self.cli.parallel_heads_env(types.SimpleNamespace(parallel_heads=None, head_input="hidden"), "qwen36", {})

    def test_no_bank_preserves_original_environment(self):
        env = {"TEMP": "existing"}
        self.assertIs(self.cli.parallel_heads_env(types.SimpleNamespace(), "glm", env), env)


if __name__ == "__main__":
    unittest.main()

import importlib.machinery
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

class MtpLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader=importlib.machinery.SourceFileLoader("coli_mtp_test",str(Path(__file__).resolve().parents[1]/"coli"))
        spec=importlib.util.spec_from_loader(loader.name,loader)
        cls.cli=importlib.util.module_from_spec(spec);loader.exec_module(cls.cli)

    def test_native_sidecar_preserves_settings_and_resolves_path(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("qwen36_mtp.json","model-mtp.safetensors"):(Path(directory)/name).write_bytes(b"fixture")
            a=types.SimpleNamespace(mtp_dir=directory,mtp_drafts=3,gpu="none")
            env=self.cli.sequential_mtp_env(a,"qwen36",{"OMP_NUM_THREADS":"8"})
            self.assertEqual(env["Q36_MTP"],str(Path(directory).resolve()))
            self.assertEqual(env["Q36_MTP_DRAFTS"],"3");self.assertEqual(env["OMP_NUM_THREADS"],"8")

    def test_unsupported_family_conflicting_heads_and_device_refuse(self):
        for arch in ("glm","qwen38","olmoe","camera_upscale"):
            with self.subTest(arch=arch),self.assertRaisesRegex(ValueError,"requires qwen36"):
                self.cli.sequential_mtp_env(types.SimpleNamespace(mtp_dir="side"),arch,{})
        with self.assertRaisesRegex(ValueError,"choose one"):
            self.cli.sequential_mtp_env(types.SimpleNamespace(mtp_dir="side"),"qwen36",{"COLI_STRATA_HEADS":"bank"})
        with self.assertRaisesRegex(ValueError,"requires CPU"):
            self.cli.sequential_mtp_env(types.SimpleNamespace(mtp_dir="side",gpu="auto"),"qwen36",{})

    def test_absent_incomplete_and_unbound_depth_do_not_launch(self):
        with self.assertRaisesRegex(ValueError,"requires --mtp-dir"):
            self.cli.sequential_mtp_env(types.SimpleNamespace(mtp_drafts=2),"qwen36",{})
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,"converted metadata"):
            self.cli.sequential_mtp_env(types.SimpleNamespace(mtp_dir=directory),"qwen36",{})

    def test_no_mtp_leaves_existing_environment_untouched(self):
        env={"OTHER":"existing"}
        self.assertIs(self.cli.sequential_mtp_env(types.SimpleNamespace(),"glm",env),env)

    def test_strata_window_default_and_explicit_depth_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("qwen36_mtp.json","model-mtp.safetensors"):(Path(directory)/name).write_bytes(b"fixture")
            a=types.SimpleNamespace(mtp_dir=directory,gpu="none")
            self.assertEqual(self.cli.sequential_mtp_env(a,"qwen36",{})["Q36_MTP_DRAFTS"],"3")
            self.assertEqual(self.cli.sequential_mtp_env(a,"qwen36",{"Q36_MTP_DRAFTS":"2"})["Q36_MTP_DRAFTS"],"2")
            a.mtp_drafts=1
            self.assertEqual(self.cli.sequential_mtp_env(a,"qwen36",{"Q36_MTP_DRAFTS":"2"})["Q36_MTP_DRAFTS"],"1")

if __name__=="__main__":unittest.main()

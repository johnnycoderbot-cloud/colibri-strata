"""Authentic-shape tiny fixtures for immutable, opt-in MTP sidecar export."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import load_file, save_file
    converter = importlib.import_module("tools.convert_qwen36")
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAVE_DEPS, "MTP converter needs torch and safetensors")
class MtpConverterTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.root = Path(self.home.name)
        self.source = self.root / "source"; self.source.mkdir()
        self.native = self.root / "native"; self.native.mkdir()
        self.output = self.root / "mtp"
        self.text = {"model_type": "qwen3_5_text", "hidden_size": 8, "vocab_size": 16,
            "intermediate_size": 12, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "num_hidden_layers": 1, "max_position_embeddings": 1024,
            "layer_types": ["full_attention"], "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False, "tie_word_embeddings": True,
            "attention_bias": False, "attn_output_gate": True, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "partial_rotary_factor": 0.5, "mrope_section": [1, 0, 0],
                                "mrope_interleaved": True}}
        self.config = {"model_type": "qwen3_5", "text_config": self.text, "tie_word_embeddings": True}
        self.mtp_shapes = {"mtp.fc.weight": (8, 16), "mtp.norm.weight": (8,),
            "mtp.pre_fc_norm_embedding.weight": (8,), "mtp.pre_fc_norm_hidden.weight": (8,),
            "mtp.layers.0.input_layernorm.weight": (8,), "mtp.layers.0.post_attention_layernorm.weight": (8,),
            "mtp.layers.0.self_attn.q_proj.weight": (16, 8), "mtp.layers.0.self_attn.k_proj.weight": (4, 8),
            "mtp.layers.0.self_attn.v_proj.weight": (4, 8), "mtp.layers.0.self_attn.o_proj.weight": (8, 8),
            "mtp.layers.0.self_attn.q_norm.weight": (4,), "mtp.layers.0.self_attn.k_norm.weight": (4,),
            "mtp.layers.0.mlp.gate_proj.weight": (12, 8), "mtp.layers.0.mlp.up_proj.weight": (12, 8),
            "mtp.layers.0.mlp.down_proj.weight": (8, 12)}
        base = {"model.embed_tokens.weight": (16, 8), "model.norm.weight": (8,)}
        for name, shape in self.mtp_shapes.items():
            if name.startswith("mtp.layers.0."):
                base[name.replace("mtp.layers.0.", "model.layers.0.")] = shape
        self.tensors = {}
        for index, (name, shape) in enumerate({**base, **self.mtp_shapes}.items()):
            n = 1
            for dim in shape: n *= dim
            self.tensors[name] = ((torch.arange(n, dtype=torch.float32) % 17 - 8) / 32 + index / 256).reshape(shape).to(torch.bfloat16)
        self.write_source()
        self.write_native()

    def write_source(self):
        (self.source / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.source / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"BPE","vocab":{"x":0}}}', encoding="utf-8")
        save_file(self.tensors, str(self.source / "model.safetensors"))
        (self.source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            name: "model.safetensors" for name in self.tensors}}), encoding="utf-8")

    def write_native(self):
        shutil.copyfile(self.source / "config.json", self.native / "config.hf.json")
        shutil.copyfile(self.source / "tokenizer.json", self.native / "tokenizer.json")
        (self.native / "config.json").write_text(json.dumps(self.text), encoding="utf-8")
        meta = {"hidden": 8, "n_layers": 1, "n_active": 1, "num_experts": 0, "shared_inter": 12,
            "rms_eps": 1e-6, "q_heads": 2, "kv_heads": 1, "q_head_dim": 8, "k_head_dim": 4,
            "v_head_dim": 4, "o_in": 8, "head_dim": 4, "rope_theta": 10000,
            "partial_rotary_factor": 0.5, "zero_centered_norms": True, "ebits": 8}
        (self.native / "qwen36_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        save_file({"model.embed_tokens.weight": self.tensors["model.embed_tokens.weight"].half().clone(),
                   "lm_head.weight": self.tensors["model.embed_tokens.weight"].half().clone(),
                   "model.norm.weight": self.tensors["model.norm.weight"].half().clone()},
                  str(self.native / "model-globals.safetensors"))
        save_file({name: value.half().clone() for name, value in self.tensors.items() if name.startswith("model.layers.")},
                  str(self.native / "model-00000.safetensors"))

    def snapshot(self):
        return {str(path.relative_to(self.root)): digest(path) for directory in (self.source, self.native)
                for path in directory.iterdir() if path.is_file() and not path.is_symlink()}

    def export(self, output=None, precision=None, revision=None):
        return converter.export_mtp_sidecar(self.source, self.native, output or self.output,
                    precision or converter.MTP_NATIVE_PRECISION, revision)

    def refused(self, regex):
        before = self.snapshot()
        with self.assertRaisesRegex((ValueError, OSError), regex): self.export()
        self.assertFalse(self.output.exists()); self.assertEqual(before, self.snapshot())

    def test_raw_bfloat_values_norms_names_identity_and_immutability(self):
        before = self.snapshot(); report = self.export(revision="a" * 40)
        self.assertEqual(before, self.snapshot())
        self.assertEqual({p.name for p in self.output.iterdir()}, {"model-mtp.safetensors", "qwen36_mtp.json"})
        exported = load_file(str(self.output / "model-mtp.safetensors"))
        self.assertEqual(set(exported), set(self.mtp_shapes))
        for name, tensor in exported.items():
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tuple(tensor.shape), self.mtp_shapes[name])
            self.assertTrue(torch.equal(tensor, self.tensors[name].float()), name)
        self.assertEqual(report["norm_convention"], "qwen_rmsnorm_plus_one")
        self.assertEqual(report["fc_order"], ["embedding", "target_hidden"])
        self.assertEqual(report["target_hidden_tap"], "post_final_norm")
        self.assertEqual(report["source"]["revision"], "a" * 40)
        self.assertEqual(report["sidecar"]["sha256"], digest(self.output / "model-mtp.safetensors"))
        self.assertLessEqual((self.output / "qwen36_mtp.json").stat().st_size, 65536)
        from tools.export_native_heads import fingerprint_model
        self.assertEqual(report["native"]["fingerprint"], fingerprint_model(self.native, converter.MTP_NATIVE_PRECISION).hex())
        self.assertEqual(report, json.loads((self.output / "qwen36_mtp.json").read_text()))
        self.assertTrue(report["shared_embedding"] and report["shared_lm_head"])

    def test_existing_directory_and_existing_sidecars_are_never_overwritten(self):
        self.output.mkdir(); sentinel = self.output / "model-mtp.safetensors"; sentinel.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "new.*overwrite"): self.export()
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_output_cannot_be_inside_source_or_native(self):
        before = self.snapshot()
        for directory in (self.source, self.native):
            for output in (directory, directory / "sidecar"):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                    self.export(output)
        self.assertEqual(before, self.snapshot())

    def test_unknown_and_incomplete_tensors_refuse_before_directory_creation(self):
        name = "mtp.layers.0.self_attn.q_proj.weight"
        original = self.tensors.pop(name); self.write_source(); self.refused("incomplete")
        self.tensors[name] = original
        self.tensors["mtp.layers.0.unrecognized.weight"] = torch.zeros(8, dtype=torch.bfloat16)
        self.write_source(); self.refused("extra")

    def test_projection_and_norm_shapes_and_nonfloat_dtype_are_checked(self):
        names = ("mtp.fc.weight", "mtp.layers.0.self_attn.q_norm.weight", "mtp.pre_fc_norm_hidden.weight")
        for name in names:
            with self.subTest(name=name):
                original = self.tensors[name]
                self.tensors[name] = torch.zeros(3, dtype=torch.bfloat16); self.write_source()
                self.refused("shape/dtype"); self.tensors[name] = original
        self.tensors["mtp.norm.weight"] = torch.zeros(8, dtype=torch.int8)
        self.write_source(); self.refused("shape/dtype")

    def test_nonfinite_raw_weight_refuses_export(self):
        self.tensors["mtp.norm.weight"][0] = float("nan"); self.write_source()
        self.refused("nonfinite MTP tensor")

    def test_declared_tensor_geometry_is_bounded_before_loading(self):
        self.text["hidden_size"] = 32768; self.write_source()
        self.refused("exceeds 512 MiB")

    def test_shared_embedding_and_native_head_dimensions_must_match(self):
        original = self.tensors["model.embed_tokens.weight"]
        self.tensors["model.embed_tokens.weight"] = torch.zeros(15, 8, dtype=torch.bfloat16)
        self.write_source(); self.refused("shared target tensor shape")
        self.tensors["model.embed_tokens.weight"] = original; self.write_source()
        path = self.native / "model-globals.safetensors"
        values = load_file(str(path)); values["lm_head.weight"] = torch.zeros(15, 8, dtype=torch.float16)
        save_file(values, str(path)); self.refused("native shared tensor shape")

    def test_moe_multicell_dedicated_and_rope_geometry_refuse(self):
        cases = (("num_experts", 4, "MoE"), ("mtp_num_hidden_layers", 2, "one pretrained"),
                 ("mtp_num_hidden_layers", True, "one pretrained"),
                 ("mtp_use_dedicated_embeddings", True, "dedicated"),
                 ("num_key_value_heads", 3, "heads"), ("hidden_size", True, "hidden_size"))
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                old = self.text.get(key); present = key in self.text
                self.text[key] = value; self.write_source(); self.refused(message)
                if present: self.text[key] = old
                else: self.text.pop(key)
        self.text["rope_parameters"]["rope_type"] = "yarn"; self.write_source(); self.refused("rope scaling")

    def test_native_config_tokenizer_geometry_and_norm_mismatches_refuse(self):
        original = (self.native / "tokenizer.json").read_bytes()
        (self.native / "tokenizer.json").write_bytes(original + b" ")
        self.refused("tokenizer bytes")
        (self.native / "tokenizer.json").write_bytes(original)
        path = self.native / "qwen36_meta.json"; original = path.read_bytes(); meta = json.loads(original)
        for key, value in (("hidden", 7), ("q_head_dim", 4), ("zero_centered_norms", False)):
            with self.subTest(key=key):
                changed = {**meta, key: value}; path.write_text(json.dumps(changed)); self.refused("geometry/norm")
        path.write_bytes(original)
        (self.native / "config.hf.json").write_text('{}'); self.refused("checkpoint config")

    def test_source_index_mismatch_traversal_and_duplicates_refuse(self):
        path = self.source / "model.safetensors.index.json"; original = path.read_bytes()
        index = json.loads(original); name = "mtp.norm.weight"
        index["weight_map"][name] = "../native/model-globals.safetensors"
        path.write_text(json.dumps(index)); self.refused("unsafe")
        index = json.loads(original); index["weight_map"].pop(name)
        path.write_text(json.dumps(index)); self.refused("index/shard mismatch")
        path.write_text('{"weight_map":{},"weight_map":{}}'); self.refused("duplicate")

    def test_source_without_index_is_supported(self):
        (self.source / "model.safetensors.index.json").unlink()
        report = self.export(); self.assertEqual(len(report["tensors"]), 15)

    def test_precision_revision_and_nonfinite_duplicate_json_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "precision/tap"): self.export(precision="qwen36:unqualified")
        with self.assertRaisesRegex(ValueError, "commit SHA"): self.export(revision="moving-main")
        path = self.source / "config.json"
        path.write_text('{"text_config":NaN}'); self.refused("nonfinite")
        path.write_text('{"text_config":{},"text_config":{}}'); self.refused("duplicate")

    def test_output_write_failure_cleans_only_new_export_files(self):
        original_open = Path.open; before = self.snapshot()
        def failing_open(path, *args, **kwargs):
            if path.name == "qwen36_mtp.json" and args and args[0] == "xb":
                raise OSError("injected metadata write refusal")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", failing_open):
            with self.assertRaisesRegex(OSError, "injected"): self.export()
        self.assertFalse(self.output.exists()); self.assertEqual(before, self.snapshot())

    def test_default_conversion_still_explicitly_skips_mtp(self):
        target = self.root / "regular-conversion"
        result = subprocess.run([sys.executable, converter.__file__, "--model", str(self.source),
                    "--out", str(target), "--ebits", "8", "--no-readme"], capture_output=True,
                    text=True, encoding="utf-8", timeout=60,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skipping 15 `mtp.*`", result.stdout)
        self.assertFalse((target / "model-mtp.safetensors").exists())
        self.assertFalse((target / "qwen36_mtp.json").exists())

    def test_cli_requires_explicit_external_target_and_precision(self):
        result = subprocess.run([sys.executable, converter.__file__, "--mtp-only", "--model", str(self.source),
                    "--out", str(self.output)], capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 2); self.assertIn("requires local", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__": unittest.main()

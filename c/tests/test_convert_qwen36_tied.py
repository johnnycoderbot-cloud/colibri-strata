"""Tiny real safetensors conversion gates for explicit tied output weights."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import load_file, save_file
    HAVE_DEPS=True
except ImportError:
    HAVE_DEPS=False

TOOL=Path(__file__).resolve().parents[1]/"tools"/"convert_qwen36.py"


@unittest.skipUnless(HAVE_DEPS,"converter needs torch and safetensors")
class TiedConverterTest(unittest.TestCase):
    def convert(self,tie="missing",omit=(),dtype=None,explicit=False,wrapped=False):
        home=tempfile.TemporaryDirectory(); self.addCleanup(home.cleanup)
        root=Path(home.name); source=root/"source"; source.mkdir(); target=root/"converted"
        dtype=dtype or torch.float32
        config={"model_type":"qwen3_5_text","hidden_size":8,"vocab_size":6,
                "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":4,"intermediate_size":16,"layer_types":["full_attention"],
                "rms_norm_eps":1e-6,"rope_theta":10000,"num_experts":0,"num_experts_per_tok":0}
        if tie!="missing": config["tie_word_embeddings"]=tie
        if wrapped: config={"model_type":"qwen3_5","text_config":config}
        (source/"config.json").write_text(json.dumps(config),encoding="utf-8")
        tensors={"model.embed_tokens.weight":torch.arange(48,dtype=torch.float32).reshape(6,8).to(dtype)/32,
                 "model.norm.weight":torch.ones(8,dtype=dtype),
                 "model.layers.0.input_layernorm.weight":torch.ones(8,dtype=dtype),
                 "model.layers.0.post_attention_layernorm.weight":torch.ones(8,dtype=dtype),
                 "model.layers.0.self_attn.q_proj.weight":torch.zeros(16,8,dtype=dtype),
                 "model.layers.0.self_attn.k_proj.weight":torch.zeros(4,8,dtype=dtype),
                 "model.layers.0.self_attn.v_proj.weight":torch.zeros(4,8,dtype=dtype),
                 "model.layers.0.self_attn.o_proj.weight":torch.zeros(8,8,dtype=dtype),
                 "model.layers.0.self_attn.q_norm.weight":torch.ones(4,dtype=dtype),
                 "model.layers.0.self_attn.k_norm.weight":torch.ones(4,dtype=dtype),
                 "model.layers.0.mlp.gate_proj.weight":torch.zeros(16,8,dtype=dtype),
                 "model.layers.0.mlp.up_proj.weight":torch.zeros(16,8,dtype=dtype),
                 "model.layers.0.mlp.down_proj.weight":torch.zeros(8,16,dtype=dtype)}
        if explicit: tensors["lm_head.weight"]=torch.full((6,8),0.125,dtype=dtype)
        for name in omit: tensors.pop(name)
        save_file(tensors,str(source/"model.safetensors"))
        result=subprocess.run([sys.executable,str(TOOL),"--model",str(source),"--out",str(target),
                               "--ebits","8","--no-readme"],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=60,
                              env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
        return result,target,tensors

    def assert_tied(self,dtype,wrapped=False):
        result,target,original=self.convert(True,dtype=dtype,wrapped=wrapped)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        path=target/"model-globals.safetensors"; tensors=load_file(str(path))
        embed=tensors["model.embed_tokens.weight"]; head=tensors["lm_head.weight"]
        self.assertEqual(embed.dtype,torch.float16); self.assertEqual(head.dtype,torch.float16)
        self.assertTrue(torch.equal(embed,original["model.embed_tokens.weight"].half()))
        self.assertTrue(torch.equal(embed,head)); self.assertNotEqual(embed.data_ptr(),head.data_ptr())
        before=head.clone(); embed[0,0]+=1
        self.assertTrue(torch.equal(head,before),"editing embedding unexpectedly changed output head")
        with path.open("rb") as stream:
            length=struct.unpack("<Q",stream.read(8))[0]; metadata=json.loads(stream.read(length))
        self.assertNotEqual(metadata["model.embed_tokens.weight"]["data_offsets"],metadata["lm_head.weight"]["data_offsets"])

    def test_tied_float32_materializes_independent_float16_head(self): self.assert_tied(torch.float32)
    def test_already_float16_source_still_clones_storage(self): self.assert_tied(torch.float16)
    def test_nested_text_config_explicit_tie_is_supported(self): self.assert_tied(torch.float16,True)

    def test_untied_or_missing_flag_refuses_missing_head(self):
        for flag in (False,"missing",1,"true"):
            with self.subTest(flag=flag):
                result,target,_=self.convert(flag)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("expected embed_tokens, lm_head and final norm",result.stdout+result.stderr)
                self.assertFalse((target/"model-globals.safetensors").exists())

    def test_tie_never_substitutes_missing_norm_or_embedding(self):
        for name in ("model.norm.weight","model.embed_tokens.weight"):
            with self.subTest(missing=name):
                result,target,_=self.convert(True,omit=(name,))
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((target/"model-globals.safetensors").exists())

    def test_explicit_head_is_preserved_without_tie(self):
        result,target,_=self.convert(False,explicit=True,dtype=torch.float16)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        tensors=load_file(str(target/"model-globals.safetensors"))
        self.assertTrue(torch.equal(tensors["lm_head.weight"],torch.full((6,8),0.125,dtype=torch.float16)))
        self.assertFalse(torch.equal(tensors["lm_head.weight"],tensors["model.embed_tokens.weight"]))


if __name__=="__main__": unittest.main()

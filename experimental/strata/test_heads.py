"""Trace validation + offline trainer behavior; no model or network required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import heads


def header(family="qwen38"):
    return {"schema": heads.TRACE_SCHEMA, "family": family,
            "checkpoint_revision": "a" * 40, "tokenizer_sha256": "b" * 64,
            "hidden_tap": "synthetic_fixture_readout", "quantization": "fixture-f32",
            "hidden_size": 2, "vocab_size": 4}


class HeadsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "trace.jsonl"
        self.rows = [{"hidden": [1., 0.], "future_tokens": [0, 1, 2, 3]},
                     {"hidden": [0., 1.], "future_tokens": [0, 3, 1, None]}]
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, meta=None, rows=None):
        self.path.write_text("\n".join(json.dumps(x) for x in [meta or header(), *(rows or self.rows)]) + "\n",
                             encoding="utf-8")

    def test_offset_alignment_and_boundaries(self):
        _, h, labels = heads.load_trace(self.path, [2, 3, 4])
        self.assertEqual(h, [[1., 0.], [0., 1.]])
        self.assertEqual(labels, [[1, 2, 3], [3, 1, -100]])

    def test_all_text_families(self):
        for family in heads.FAMILIES:
            self.write(header(family.id))
            if family.modality == "text":
                meta, _, _ = heads.load_trace(self.path, [2])
                self.assertEqual(meta["family"], family.id)
            else:
                with self.assertRaises(ValueError):
                    heads.load_trace(self.path, [2])

    def test_identity_and_bad_data(self):
        for changes in ({"checkpoint_revision": "main"}, {"tokenizer_sha256": "wrong"},
                        {"checkpoint_revision": int("1" * 40)}, {"tokenizer_sha256": int("2" * 64)},
                        {"hidden_size": True}, {"quantization": ""}, {"hidden_tap": "x\0y"}):
            self.write(header() | changes)
            with self.assertRaises(ValueError):
                heads.load_trace(self.path, [2])
        for row in ({"hidden": [float("nan"), 0.], "future_tokens": [0, 1]},
                    {"hidden": [10**400, 0.], "future_tokens": [0, 1]},
                    {"hidden": [1., 0.], "future_tokens": [0, 4]},
                    {"hidden": [1., 0.], "future_tokens": [0, True]},
                    {"hidden": [1.], "future_tokens": [0, 1]}):
            self.write(rows=[row])
            with self.assertRaises(ValueError):
                heads.load_trace(self.path, [2])

    def test_offsets(self):
        for offsets in ([1], [2, 2], [3, 2], [10], []):
            with self.assertRaises(ValueError):
                heads.load_trace(self.path, offsets)

    def test_c_string_encoding(self):
        self.assertEqual(heads.c_string('x"\\\n'), '"x\\042\\134\\012"')
        self.assertEqual(heads.c_string("🦜"), '"\\360\\237\\246\\234"')

    def test_pack_identity_and_export(self):
        pack = {"schema": heads.PACK_SCHEMA, "identity": heads.identity(header()),
                "heads": [{"future_offset": 2, "rank": 1,
                           "down": [1., 0.], "up": [0., 1., 0., 0.], "bias": [0.] * 4}]}
        heads.validate_pack(pack, header())
        text = heads.export_c(pack)
        self.assertIn(".hidden_dim=2", text)
        self.assertIn(".future_offset=2", text)
        self.assertIn("strata_trained_checkpoint_revision", text)
        with self.assertRaises(ValueError):
            heads.validate_pack(pack, header() | {"tokenizer_sha256": "c" * 64})
        with self.assertRaises(ValueError):
            heads.export_c(pack, "x; evil")
        pack["heads"][0]["up"][0] = float("inf")
        with self.assertRaises(ValueError):
            heads.validate_pack(pack)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
    def test_training_learns_independent_offsets(self):
        pack = heads.train(self.path, [2, 3], 2, 100, 2, 0.05, 42)
        self.assertLess(pack["training"]["last_loss"], pack["training"]["first_loss"] * 0.1)
        self.assertFalse(pack["training"]["quality_validated"])
        self.assertEqual([h["future_offset"] for h in pack["heads"]], [2, 3])
        import torch
        for k, head in enumerate(pack["heads"]):
            down = torch.tensor(head["down"]).reshape(2, 2)
            up = torch.tensor(head["up"]).reshape(4, 2)
            bias = torch.tensor(head["bias"])
            for row in self.rows:
                prediction = int(torch.argmax(up @ (down @ torch.tensor(row["hidden"])) + bias))
                self.assertEqual(prediction, row["future_tokens"][k + 1])


if __name__ == "__main__":
    unittest.main()

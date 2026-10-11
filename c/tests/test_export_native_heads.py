"""Exporter/format behavior and optional independently compiled native ABI."""
import copy
import ctypes
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

CROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_native_heads", CROOT / "tools" / "export_native_heads.py")
exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)


def model(directory, width=17, vocab=9):
    root = Path(directory)
    (root / "config.json").write_text(json.dumps({"text_config": {"model_type": "qwen3_5_text", "hidden_size": width, "vocab_size": vocab}}))
    (root / "tokenizer.json").write_bytes(b"tokenizer fixture")
    (root / "z.safetensors").write_bytes(b"z tensor bytes")
    (root / "a.safetensors").write_bytes(b"a tensor bytes")
    (root / "qwen36_meta.json").write_bytes(b"native metadata")
    return root


def pack(directory, count=2, rank=17, runtime_precision="fixture-native-f32"):
    native = exporter.fingerprint_info(directory, runtime_precision)
    width, vocab = native["hidden_size"], native["vocab_size"]
    return {"schema": exporter.heads.PACK_SCHEMA,
            "identity": {"family": "qwen36", "checkpoint_revision": "a" * 40,
                         "native_fingerprint": native["fingerprint"],
                         "tokenizer_sha256": native["tokenizer_sha256"], "hidden_tap": native.get("hidden_tap", "native_final_norm"),
                         "quantization": runtime_precision, "hidden_size": width, "vocab_size": vocab},
            "heads": [{"future_offset": k + 2, "rank": rank,
                       "down": [float(r == d or (r == rank - 1 and d == 0)) for r in range(rank) for d in range(width)],
                       "up": [float(v == r) for v in range(vocab) for r in range(rank)],
                       "bias": [20. if k % 2 and v == 1 else 0. for v in range(vocab)]} for k in range(count)]}


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = model(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_fingerprint_canonical_labels_bytes_and_precision(self):
        before = exporter.fingerprint_model(self.root, "fixture-native-f32")
        (self.root / "ignored.txt").write_bytes(b"ignored")
        self.assertEqual(before, exporter.fingerprint_model(self.root, "fixture-native-f32"))
        self.assertNotEqual(before, exporter.fingerprint_model(self.root, "fixture-native-int8"))
        (self.root / "a.safetensors").rename(self.root / "b.safetensors")
        self.assertNotEqual(before, exporter.fingerprint_model(self.root, "fixture-native-f32"))
        (self.root / "b.safetensors").rename(self.root / "a.safetensors")
        (self.root / "a.safetensors").write_bytes(b"other tensor bytes")
        self.assertNotEqual(before, exporter.fingerprint_model(self.root, "fixture-native-f32"))
        with self.assertRaises(ValueError): exporter.fingerprint_model(self.root, "")

    def test_export_true_precision_shapes_and_payload_digest(self):
        p = pack(self.root)
        for bits in (4, 8, 16):
            blob = exporter.export_pack(p, self.root, "fixture-native-f32", bits, 64)
            self.assertEqual(struct.unpack_from("<4s7I", blob), (b"CSNH", 1, 96, 17, 9, 2, bits, 0))
            self.assertEqual(blob[32:64], exporter.fingerprint_model(self.root, "fixture-native-f32"))
            self.assertEqual(blob[64:96], hashlib.sha256(blob[96:]).digest())
            cursor = 96
            for index in range(2):
                offset, rank, down_bytes, up_bytes, bias_count, reserved = struct.unpack_from("<IIQQII", blob, cursor)
                self.assertEqual((offset, rank, bias_count, reserved), (index + 2, 17, 9, 0))
                cursor += 32
                down_info, down = exporter.precision.decode(blob[cursor:cursor + down_bytes]); cursor += down_bytes
                up_info, up = exporter.precision.decode(blob[cursor:cursor + up_bytes]); cursor += up_bytes
                self.assertEqual(down_info["bits"], bits); self.assertEqual(up_info["bits"], bits)
                self.assertEqual(down, p["heads"][index]["down"]); self.assertEqual(up, p["heads"][index]["up"])
                cursor += bias_count * 4
            self.assertEqual(cursor, len(blob))

    def test_rejects_identity_offsets_nonfinite_and_duplicate_json(self):
        original = pack(self.root)
        for change in ("precision", "tokenizer", "width", "offset", "nan"):
            p = copy.deepcopy(original)
            if change == "precision": p["identity"]["quantization"] = "different"
            if change == "tokenizer": p["identity"]["tokenizer_sha256"] = "b" * 64
            if change == "width": p["identity"]["hidden_size"] = 18
            if change == "offset": p["heads"][1]["future_offset"] = 4
            if change == "nan": p["heads"][1]["bias"][0] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                exporter.export_pack(p, self.root, "fixture-native-f32")
        path = self.root / "bad.json"; path.write_text('{"x":1,"x":2}')
        with self.assertRaises(ValueError): exporter._json(path)

    def test_cli_fingerprint_and_asset_overwrite_refusal(self):
        args = [sys.executable, str(CROOT / "tools" / "export_native_heads.py"), "--model-dir", str(self.root),
                "--runtime-precision", "fixture-native-f32"]
        result = subprocess.run(args + ["--fingerprint-only"], capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        self.assertEqual(info["hidden_size"], 17); self.assertEqual(info["vocab_size"], 9)
        source = self.root / "heads.json"; source.write_text(json.dumps(pack(self.root)))
        result = subprocess.run(args + ["--pack", str(source), "--output", str(self.root / "config.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        existing = self.root / "existing.snh"; existing.write_bytes(b"keep existing bank")
        result = subprocess.run(args + ["--pack", str(source), "--output", str(existing)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(existing.read_bytes(), b"keep existing bank")

    def test_augmented_tap_width_fingerprint_and_export_binding(self):
        augmented = "qwen36:DBITS=8;kernel=cpu;tap=" + exporter.AUGMENTED_TAP
        default = "qwen36:DBITS=8;kernel=cpu;tap=final-rmsnorm/v1"
        model(self.root, width=1024, vocab=248320)
        info = exporter.fingerprint_info(self.root, augmented)
        self.assertEqual((info["hidden_size"], info["native_hidden_size"], info["vocab_size"]), (2048, 1024, 248320))
        self.assertEqual(info["hidden_tap"], exporter.AUGMENTED_TAP)
        plain = exporter.fingerprint_info(self.root, default)
        self.assertEqual(plain["hidden_size"], 1024)
        self.assertNotIn("native_hidden_size", plain)
        self.assertNotEqual(info["fingerprint"], plain["fingerprint"])
        model(self.root)
        p = pack(self.root, runtime_precision=augmented)
        blob = exporter.export_pack(p, self.root, augmented)
        self.assertEqual(struct.unpack_from("<4s7I", blob)[3], 34)
        self.assertEqual(blob[32:64], exporter.fingerprint_model(self.root, augmented))
        wrong_tap = copy.deepcopy(p); wrong_tap["identity"]["hidden_tap"] = "native_final_norm"
        with self.assertRaisesRegex(ValueError, "hidden tap"):
            exporter.export_pack(wrong_tap, self.root, augmented)
        legacy = pack(self.root)
        legacy["identity"]["hidden_tap"] = exporter.AUGMENTED_TAP
        with self.assertRaisesRegex(ValueError, "augmented runtime"):
            exporter.export_pack(legacy, self.root, "fixture-native-f32")
        for descriptor in (augmented + ";kernel=cpu", "qwen36:tap=final-rmsnorm/v1;tap=" + exporter.AUGMENTED_TAP):
            with self.subTest(descriptor=descriptor), self.assertRaisesRegex(ValueError, "terminal tap"):
                exporter.fingerprint_info(self.root, descriptor)
        # A similarly named opaque field must not change the feature width.
        self.assertEqual(exporter.fingerprint_info(self.root, "qwen36:notap=" + exporter.AUGMENTED_TAP)["hidden_size"], 17)
        model(self.root, width=16385)
        with self.assertRaisesRegex(ValueError, "feature hidden_size"):
            exporter.fingerprint_info(self.root, augmented)

    def test_family_and_authoritative_checkpoint_binding(self):
        p = pack(self.root)
        wrong = copy.deepcopy(p); wrong["identity"]["family"] = "olmoe"
        with self.assertRaisesRegex(ValueError, "head family"):
            exporter.export_pack(wrong, self.root, "fixture-native-f32")
        with self.assertRaisesRegex(ValueError, "families differ"):
            exporter.fingerprint_info(self.root, "olmoe:DBITS=8")
        path = self.root / "config.json"; config = json.loads(path.read_text())
        config["_commit_hash"] = "a" * 40; path.write_text(json.dumps(config))
        p = pack(self.root)
        self.assertEqual(exporter.fingerprint_info(self.root, "fixture-native-f32")["checkpoint_revision"], "a" * 40)
        exporter.export_pack(p, self.root, "fixture-native-f32")
        wrong = copy.deepcopy(p); wrong["identity"]["checkpoint_revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "head checkpoint"):
            exporter.export_pack(wrong, self.root, "fixture-native-f32")
        config["text_config"]["_commit_hash"] = "b" * 40; path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "checkpoint revisions differ"):
            exporter.fingerprint_info(self.root, "fixture-native-f32")

    def test_requires_exact_training_native_fingerprint(self):
        p = pack(self.root)
        for fingerprint in (None, "b" * 64):
            wrong = copy.deepcopy(p)
            if fingerprint is None:
                del wrong["identity"]["native_fingerprint"]
            else:
                wrong["identity"]["native_fingerprint"] = fingerprint
            with self.subTest(fingerprint=fingerprint), self.assertRaisesRegex(ValueError, "native_fingerprint"):
                exporter.export_pack(wrong, self.root, "fixture-native-f32")
        original = exporter.fingerprint_info(self.root, "fixture-native-f32")
        (self.root / "a.safetensors").write_bytes(b"different target with identical geometry and tokenizer")
        changed = exporter.fingerprint_info(self.root, "fixture-native-f32")
        self.assertEqual(original["hidden_size"], changed["hidden_size"])
        self.assertEqual(original["vocab_size"], changed["vocab_size"])
        self.assertEqual(original["tokenizer_sha256"], changed["tokenizer_sha256"])
        with self.assertRaisesRegex(ValueError, "native_fingerprint"):
            exporter.export_pack(p, self.root, "fixture-native-f32")
        augmented = "qwen36:DBITS=8;kernel=cpu;tap=" + exporter.AUGMENTED_TAP
        model(self.root)
        wrong = pack(self.root, runtime_precision=augmented)
        wrong["identity"]["native_fingerprint"] = exporter.fingerprint_info(self.root, "fixture-native-f32")["fingerprint"]
        with self.assertRaisesRegex(ValueError, "native_fingerprint"):
            exporter.export_pack(wrong, self.root, augmented)

    def test_residual_requires_explicit_kind_and_preserves_encoded_bytes(self):
        p = pack(self.root)
        residual = {**p, "schema": exporter.RESIDUAL_SCHEMA, "kind": "logit-residual"}
        with self.assertRaises(ValueError): exporter.export_pack(residual, self.root, "fixture-native-f32")
        with self.assertRaises(ValueError): exporter.export_pack(p, self.root, "fixture-native-f32", kind="logit-residual")
        with self.assertRaises(ValueError): exporter.export_pack(p, self.root, "fixture-native-f32", kind="unknown")
        with self.assertRaises(ValueError): exporter.export_pack({**p, "kind": "logit-residual"}, self.root, "fixture-native-f32")
        for bits in (4, 8, 16):
            blob = exporter.export_pack(residual, self.root, "fixture-native-f32", bits, kind="logit-residual")
            self.assertEqual(struct.unpack_from("<4s7I", blob), (b"CSNH", 2, 96, 17, 9, 2, bits, 1))
            records = [{**head, "down": exporter.precision.encode(head["down"], bits),
                         "up": exporter.precision.encode(head["up"], bits)} for head in p["heads"]]
            borrowed = exporter.export_encoded_residual(p["identity"], records, self.root, "fixture-native-f32", bits)
            self.assertEqual(blob, borrowed)
        records = [{**head, "down": exporter.precision.encode(head["down"], 4),
                     "up": exporter.precision.encode(head["up"], 4)} for head in p["heads"]]
        for change in ("not-object", "float-offset", "missing-down", "missing-up", "truncated", "padding", "rank", "bias", "precision"):
            wrong = copy.deepcopy(records); requested = 4
            if change == "not-object": wrong[0] = None
            if change == "float-offset": wrong[0]["future_offset"] = 2.0
            if change == "missing-down": del wrong[0]["down"]
            if change == "missing-up": del wrong[0]["up"]
            if change == "truncated": wrong[1]["up"] = wrong[1]["up"][:-1]
            if change == "padding":
                bad = bytearray(wrong[0]["down"]); bad[-1] |= 16; wrong[0]["down"] = bytes(bad)
            if change == "rank": wrong[0]["rank"] = 2
            if change == "bias": wrong[1]["bias"][0] = float("nan")
            if change == "precision": requested = 8
            with self.subTest(change=change), self.assertRaises(ValueError):
                exporter.export_encoded_residual(p["identity"], wrong, self.root, "fixture-native-f32", requested)

    def test_cli_residual_header(self):
        p = pack(self.root); p.update(schema=exporter.RESIDUAL_SCHEMA, kind="logit-residual")
        source = self.root / "residual.json"; source.write_text(json.dumps(p))
        output = self.root / "residual.snh"
        command = [sys.executable, str(CROOT / "tools/export_native_heads.py"), "--model-dir", str(self.root),
                   "--runtime-precision", "fixture-native-f32", "--pack", str(source), "--output", str(output),
                   "--bank-kind", "logit-residual", "--bits", "8"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["version"], 2)
        self.assertEqual(struct.unpack_from("<I", output.read_bytes(), 28)[0], 1)


@unittest.skipUnless(shutil.which("gcc"), "optional GCC native ABI interop compiler unavailable")
class NativeInteropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        suffix = ".dll" if sys.platform == "win32" else ".so"
        library = Path(cls.build.name) / ("native_heads" + suffix)
        subprocess.run([shutil.which("gcc"), "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-fopenmp",
                        "-shared", "-fPIC", str(CROOT / "strata_native_heads.c"), "-lm", "-o", str(library)], check=True, capture_output=True)
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.sn_fingerprint_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8)]
        cls.lib.sn_fingerprint_model.restype = ctypes.c_int
        cls.lib.sn_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        cls.lib.sn_open.restype = ctypes.c_int
        cls.lib.sn_predict_sized.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                           ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_int]
        cls.lib.sn_predict_sized.restype = ctypes.c_int
        cls.lib.sn_bank_kind.argtypes = [ctypes.c_void_p]
        cls.lib.sn_bank_kind.restype = ctypes.c_int
        cls.lib.sn_predict_residual_sized.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                                     ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                                     ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_int]
        cls.lib.sn_predict_residual_sized.restype = ctypes.c_int
        cls.lib.sn_close.argtypes = [ctypes.c_void_p]
        cls.lib.sn_close.restype = None

    @classmethod
    def tearDownClass(cls): cls.build.cleanup()

    def test_cross_language_fingerprint_and_all_precision_max_head_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory); p = pack(root, count=8)
            expected = exporter.fingerprint_model(root, "fixture-native-f32")
            fp = (ctypes.c_uint8 * 32)()
            self.assertEqual(self.lib.sn_fingerprint_model(str(root).encode(), b"fixture-native-f32", fp), 0)
            self.assertEqual(bytes(fp), expected)
            hidden = (ctypes.c_float * 17)(*range(17))
            for bits in (4, 8, 16):
                path = root / "bank.snh"; path.write_bytes(exporter.export_pack(p, root, "fixture-native-f32", bits))
                bank = ctypes.c_void_p()
                self.assertEqual(self.lib.sn_open(str(path).encode(), fp, 17, 9, ctypes.byref(bank)), 0)
                try:
                    path.unlink()  # Decode-once ownership, not per-call file reads.
                    for parallel in (0, 1):
                        tokens = (ctypes.c_int * 8)(*([-1] * 8))
                        self.assertEqual(self.lib.sn_predict_sized(bank, hidden, 17, tokens, 8, parallel), 0)
                        self.assertEqual(list(tokens), [8, 1, 8, 1, 8, 1, 8, 1])
                finally: self.lib.sn_close(bank)

    def test_nontrivial_decoded_reference_predictions(self):
        rng = random.Random(1977)
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory, width=11, vocab=23); p = pack(root, count=3, rank=5)
            for head in p["heads"]:
                for key in ("down", "up", "bias"):
                    head[key] = [rng.uniform(-0.8, 0.8) for _ in head[key]]
            fp = (ctypes.c_uint8 * 32).from_buffer_copy(exporter.fingerprint_model(root, "fixture-native-f32"))
            inputs = [[rng.uniform(-2, 2) for _ in range(11)] for _ in range(40)]
            for bits in (4, 8, 16):
                path = root / "bank.snh"; path.write_bytes(exporter.export_pack(p, root, "fixture-native-f32", bits))
                bank = ctypes.c_void_p()
                self.assertEqual(self.lib.sn_open(str(path).encode(), fp, 11, 23, ctypes.byref(bank)), 0)
                try:
                    for row in inputs:
                        expected = [max(range(23), key=lambda v: logits[v]) for logits in exporter.precision.reference(p, row, bits)]
                        hidden = (ctypes.c_float * 11)(*row)
                        for parallel in (0, 1):
                            tokens = (ctypes.c_int * 3)(-1, -1, -1)
                            self.assertEqual(self.lib.sn_predict_sized(bank, hidden, 11, tokens, 3, parallel), 0)
                            self.assertEqual(list(tokens), expected)
                finally: self.lib.sn_close(bank)

    def test_augmented_features_and_model_tap_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            descriptor = "qwen36:DBITS=8;kernel=cpu;tap=" + exporter.AUGMENTED_TAP
            p = pack(root, count=1, rank=1, runtime_precision=descriptor)
            head = p["heads"][0]
            head["down"] = [0.] * 33 + [1.]
            head["up"] = [-1.] * 8 + [1.]
            head["bias"] = [0.] * 9
            path = root / "bank.snh"; path.write_bytes(exporter.export_pack(p, root, descriptor))
            fp = (ctypes.c_uint8 * 32).from_buffer_copy(exporter.fingerprint_model(root, descriptor))
            native_fp = (ctypes.c_uint8 * 32)()
            self.assertEqual(self.lib.sn_fingerprint_model(str(root).encode(), descriptor.encode(), native_fp), 0)
            self.assertEqual(bytes(native_fp), bytes(fp))
            wrong_tap = exporter.fingerprint_model(root, "qwen36:DBITS=8;kernel=cpu;tap=final-rmsnorm/v1")
            (root / "a.safetensors").write_bytes(b"different native tensors")
            wrong_model = exporter.fingerprint_model(root, descriptor)
            for wrong in (wrong_tap, wrong_model):
                bank = ctypes.c_void_p()
                self.assertEqual(self.lib.sn_open(str(path).encode(), (ctypes.c_uint8 * 32).from_buffer_copy(wrong), 34, 9, ctypes.byref(bank)), 3)
                self.assertFalse(bank.value)
            bank = ctypes.c_void_p()
            self.assertEqual(self.lib.sn_open(str(path).encode(), fp, 34, 9, ctypes.byref(bank)), 0)
            try:
                for value, expected in ((1., 8), (-1., 0)):
                    features = (ctypes.c_float * 34)(*([0.] * 33 + [value]))
                    for parallel in (0, 1):
                        tokens = (ctypes.c_int * 1)(-1)
                        self.assertEqual(self.lib.sn_predict_sized(bank, features, 34, tokens, 1, parallel), 0)
                        self.assertEqual(tokens[0], expected)
            finally: self.lib.sn_close(bank)

    def test_residual_reference_modes_kinds_and_concurrent_inputs(self):
        rng = random.Random(404)
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory, width=11, vocab=23); p = pack(root, count=8, rank=5)
            p.update(schema=exporter.RESIDUAL_SCHEMA, kind="logit-residual")
            for head in p["heads"]:
                for key in ("down", "up", "bias"):
                    head[key] = [rng.uniform(-.8, .8) for _ in head[key]]
            fp = (ctypes.c_uint8 * 32).from_buffer_copy(exporter.fingerprint_model(root, "fixture-native-f32"))
            rows = [[exporter.precision.f32(rng.uniform(-2, 2)) for _ in range(11)] for _ in range(12)]
            bases = [[exporter.precision.f32(rng.uniform(-1, 1)) for _ in range(23)] for _ in rows]
            for bits in (4, 8, 16):
                blob = exporter.export_pack(p, root, "fixture-native-f32", bits, kind="logit-residual")
                path = root / "bank.snh"; path.write_bytes(blob); bank = ctypes.c_void_p()
                self.assertEqual(self.lib.sn_open(str(path).encode(), fp, 11, 23, ctypes.byref(bank)), 0)
                try:
                    self.assertEqual(self.lib.sn_bank_kind(bank), 1)
                    decoded = [{key: exporter.precision.decode(exporter.precision.encode(head[key], bits))[1]
                                for key in ("down", "up")} for head in p["heads"]]
                    def check(index):
                        row, base = rows[index], bases[index]
                        expected = []
                        for head, weights in zip(p["heads"], decoded):
                            latent = [exporter.precision.f32(sum(weights["down"][r*11+d]*row[d] for d in range(11))) for r in range(5)]
                            scores = [exporter.precision.f32(sum(weights["up"][v*5+r]*latent[r] for r in range(5))+
                                                            exporter.precision.f32(head["bias"][v])+base[v]) for v in range(23)]
                            expected.append(max(range(23), key=lambda v: scores[v]))
                        x = (ctypes.c_float*11)(*row); b = (ctypes.c_float*23)(*base)
                        before = bytes(b)
                        for parallel in (0, 1):
                            out = (ctypes.c_int*8)(*([-9]*8))
                            self.assertEqual(self.lib.sn_predict_residual_sized(bank, x, 11, b, 23, out, 8, parallel), 0)
                            self.assertEqual(list(out), expected); self.assertEqual(bytes(b), before)
                            self.assertEqual(self.lib.sn_predict_sized(bank, x, 11, out, 8, parallel), 1)
                            self.assertEqual(list(out), expected)
                        b[22] = float("nan"); out = (ctypes.c_int*8)(*([-9]*8))
                        self.assertEqual(self.lib.sn_predict_residual_sized(bank, x, 11, b, 23, out, 8, 1), 6)
                        self.assertEqual(list(out), [-9]*8)
                    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(check, range(len(rows))))
                finally: self.lib.sn_close(bank)
                for version, kind in ((1, 1), (2, 0), (2, 2), (3, 1)):
                    bad = bytearray(blob); struct.pack_into("<I", bad, 4, version); struct.pack_into("<I", bad, 28, kind)
                    path.write_bytes(bad); rejected = ctypes.c_void_p()
                    self.assertEqual(self.lib.sn_open(str(path).encode(), fp, 11, 23, ctypes.byref(rejected)), 1)
                    self.assertFalse(rejected.value)


if __name__ == "__main__": unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Convert Qwen3.6-35B-A3B (or any Qwen3.5/3.6 MoE) HF checkpoint -> colibri container.

Phase-2 converter: it converts ALL layers (GATED-ATTENTION full_attention AND the
Gated DeltaNet linear_attention layers). Every layer also carries its MoE/MLP block.
DeltaNet layers are stored under `model.layers.{i}.linear_attn.*` (in_proj_qkv/z/b/a,
conv1d.weight, dt_bias, A_log, norm.weight, out_proj). The engine implements the
recurrent gated-delta-rule for these layers; in Phase 1 they were skipped as identity.

Design notes (must stay in sync with c/qwen36.c):
  * The colibri container is just a directory of safetensors shards (+ config.json +
    qwen36_meta.json). The engine reads them lazily via st_init, paging expert weights
    from disk -- that is the whole "run a huge model in little RAM" trick.
  * Expert weights are stored per-expert as `model.layers.{a}.mlp.experts.{e}.merged_weight`
    + `.qs` (f32 scales). For --ebits >= 5 the merged_weight is int8 (layout [gate|up|down]);
    for --ebits <= 4 it is TRUE 4-bit packed uint8 (2 elements/byte, half size) -- see
    pack_int4(). The qs layout (per-row f32 scales) is identical either way. This matches
    what c/qwen36.c's load_expert_merged expects: it detects int4 by ON-DISK SIZE (N/2 bytes)
    and unpacks in-place to int8, so the rest of the MoE path is unchanged.
  * --down-bits 8 (with --ebits <= 4) is the MIXED layout: gate and up stay packed int4,
    down_proj is int8 (--down-gs groups along its input, 0 = per row). One merged_weight per
    expert still: uint8 [gate int4 packed | up int4 packed | down int8], 2*inter*hidden bytes
    (int4: 1.5x, int8: 3x), qs = [gate scales | up scales | down scales]. Written to meta as
    expert_down_bits / expert_down_gs. Q4_K_M keeps down/output in Q6_K; this is the
    experiment #1370 asked for, with the engine reading each matrix in its own format.
  * Attention + router + shared-expert + norms stay f16 (they are tiny vs experts).
  * Real Qwen3.6 is a vision-language checkpoint: config dims live under `text_config`,
    and weight keys are prefixed `model.language_model.`. Both are handled transparently.
  * The fused expert tensors `mlp.experts.gate_up_proj` / `down_proj` are split per expert
    into the merged_weight layout (gate_up = [gate; up] along dim 0).
  * Every tensor name is classified against tools/qwen36_tensor_kinds.py before the first
    shard is read. The `mtp.*` head and the `visual.*` tower are skipped ON PURPOSE and
    reported with a count; a name the contract does not know stops the conversion. A
    converter that silently drops what it does not recognise produces a container that
    loads and is quietly missing a tensor (GLM-5.3 precedent, #1045).
  * Head dims are derived from the actual weight shapes (authoritative), not from config
    heuristics, because Qwen3.6's qk split (q_head_dim=512, k/v_head_dim=256, rope over
    256 dims) does not match naive head_dim=hidden/n_heads.

Usage (cloud, e.g. Colab free CPU):
  python tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B --out ./qwen36_i4 --ebits 4 \\
      --low-disk --stream-upload --upload-repo minne100/qwen36-35b-a3b-colibri-i4

Usage (local tiny model for end-to-end testing):
  python tools/convert_qwen36.py --model ../qwen36_tiny --out ../qwen36_tiny_i4 --ebits 8

Usage (mixed: int4 gs64 gate/up, int8 down -- the #1370 experiment):
  python tools/convert_qwen36.py --model <hf> --out ./qwen36_i4_gs64_d8 --ebits 4 --gs 64 --down-bits 8
"""

import argparse, hashlib, json, math, os, stat, struct, sys, tempfile
from pathlib import Path

# Windows: force UTF-8 output
if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
    from safetensors.numpy import safe_open as safe_open_np
    from safetensors.torch import safe_open as safe_open_pt
    from safetensors.torch import save_file, load_file
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install torch safetensors")


def quantize_row(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise symmetric quantization to `bits` (2..8). Storage is int8 regardless of
    bits (engine dequantizes q*scale); for bits<8 values are confined to a smaller range."""
    qmax = (1 << (bits - 1)) - 1
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / qmax
    q = (w_f32 / scales).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scales.squeeze(1)


def pack_int4(q: "torch.Tensor") -> "torch.Tensor":
    """Pack a 1D int-tensor (values already confined to the signed-4-bit range [-8,7])
    into a uint8 blob, 2 values per byte (LOW nibble = element 2k, HIGH nibble = element 2k+1).

    Convention (kept in sync with the c/qwen36.c int4 dequant path, TODO):
      byte k  = (signed4(element[2k+1]) << 4) | (signed4(element[2k]) & 0xF)
    Each nibble is a SIGNED 4-bit integer in [-8,7]; dequant = nibble * scale[row].
    The whole expert merged blob (gate||up||down, flattened) is packed as one contiguous
    uint8 array, so its length is exactly N/2 (N = element count, always even here).
    """
    q = q.to(torch.int16).clamp(-8, 7)
    n = q.numel()
    if n & 1:
        q = torch.cat([q, torch.zeros(1, dtype=torch.int16)])
    lo = (q[0::2] & 0xF).to(torch.uint8)
    hi = ((q[1::2] & 0xF) << 4).to(torch.uint8)
    return (lo | hi).contiguous()


def quantize_row_grouped(w: "torch.Tensor", bits: int, gs: int):
    """Group-wise symmetric quantization: one f32 scale per `gs` input elements
    per row (like GLM's gs64 containers). Returns (q int8 [O,I], scales f32
    [O, ceil(I/gs)] flattened row-major)."""
    qmax = (1 << (bits - 1)) - 1
    O, I = w.shape[0], w.reshape(w.shape[0], -1).shape[1]
    w_f32 = w.reshape(O, -1).float()
    pad = (-I) % gs
    if pad:
        w_f32 = torch.nn.functional.pad(w_f32, (0, pad))
    ng = w_f32.shape[1] // gs
    g = w_f32.view(O, ng, gs)
    scales = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
    q = (g / scales).round().clamp(-qmax - 1, qmax).to(torch.int8).view(O, -1)[:, :I]
    return q, scales.view(O, ng)


def make_merged(gate, up, down, ebits, gs=0, down_bits=0, down_gs=0):
    gsz = gs   # local `gs` is rebound to the gate scales below -- keep the group size safe
    """gate/up: [inter, H]; down: [H, inter] (torch, any fp).
    -> (merged_weight, qs f32 1D).

    Storage format depends on ebits:
      * ebits >= 5 : merged_weight is int8  (1 byte / element), as before.
      * ebits <= 4 : merged_weight is TRUE 4-bit packed uint8 (2 elements / byte,
                     half the size) -- see pack_int4 for the nibble convention.
    The engine (c/qwen36.c) currently reads ebits>=5 (int8); the int4 path is WIP.
    qs (per-row f32 scales) is identical in both cases.
    """
    def q(t, bits, group):
        if group:
            qt, s = quantize_row_grouped(t, bits, group)   # [O, ng] scales
            return qt, s.reshape(-1)
        qt, s = quantize_row(t.reshape(t.shape[0], -1), bits)  # rows along dim0
        return qt, s
    gq, gs = q(gate, ebits, gsz)
    uq, us = q(up, ebits, gsz)
    mixed = bool(down_bits) and down_bits != ebits
    if mixed:
        if not (ebits <= 4 and down_bits >= 5):
            raise ValueError(f"mixed layout needs --ebits <= 4 and --down-bits >= 5 (got {ebits}/{down_bits})")
        # gate|up packed int4, down int8 in the same uint8 slab: 2*inter*hidden bytes,
        # which is neither the int4 (1.5x) nor the int8 (3x) size -- that is how the
        # engine tells the three layouts apart, from the bytes, not from meta.
        dq, ds = q(down, down_bits, down_gs)
        gu = pack_int4(torch.cat([gq.flatten(), uq.flatten()]).contiguous())
        mw = torch.cat([gu, dq.flatten().to(torch.int8).view(torch.uint8)]).contiguous()
    else:
        dq, ds = q(down, ebits, gsz)
        mw_i8 = torch.cat([gq.flatten(), uq.flatten(), dq.flatten()]).contiguous()
        if ebits <= 4:
            mw = pack_int4(mw_i8)              # uint8, 2x4-bit per byte -> HALF size (true int4)
        else:
            mw = mw_i8.to(torch.int8)          # int8 storage (ebits 5..8)
    qs = torch.cat([gs, us, ds]).contiguous().float()
    return mw, qs


def _unpack_int4(packed: "torch.Tensor") -> "torch.Tensor":
    """Inverse of pack_int4: uint8 blob -> signed int8 values (HIGH nibble = element 2k+1)."""
    b = packed.to(torch.uint8)
    lo = (b & 0xF).to(torch.int8)
    hi = ((b >> 4) & 0xF).to(torch.int8)
    # sign-extend 4-bit -> 8-bit
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=1).flatten()


def _selftest():
    """Round-trip check for the true-int4 packing used by --ebits<=4."""
    print("=== int4 pack/unpack selftest ===")
    ok = True
    for ebits in (2, 3, 4):
        # small synthetic expert: gate[inter,H] up[inter,H] down[H,inter]
        inter, H = 8, 16
        g = torch.randn(inter, H) * 3
        u = torch.randn(inter, H) * 3
        d = torch.randn(H, inter) * 3
        mw, qs = make_merged(g, u, d, ebits)
        # serialize to safetensors + reload (exercises the real dtype path)
        td = tempfile.mkdtemp()
        save_file({"merged_weight": mw, "qs": qs}, os.path.join(td, "e0.safetensors"))
        back = load_file(os.path.join(td, "e0.safetensors"))
        mw_r = back["merged_weight"]
        # rebuild the int8 reference exactly as make_merged did (pre-pack)
        def qref(t):
            qt, _ = quantize_row(t.reshape(t.shape[0], -1), ebits)
            return qt.flatten()
        ref = torch.cat([qref(g), qref(u), qref(d)]).contiguous().to(torch.int8)
        if mw_r.dtype != torch.uint8:
            print(f"  ebits={ebits}: FAIL dtype={mw_r.dtype} (expected uint8)")
            ok = False
            continue
        rec = _unpack_int4(mw_r).to(torch.int8)[:ref.numel()]
        maxerr = (rec.int() - ref.int()).abs().max().item()
        int8_bytes = ref.numel()
        int4_bytes = mw_r.numel()
        exact = bool(maxerr == 0)
        halved = bool(int4_bytes * 2 == int8_bytes)
        print(f"  ebits={ebits}: maxerr={maxerr} exact={exact} "
              f"int4_bytes={int4_bytes} int8_bytes={int8_bytes} halved={halved}")
        ok = ok and exact and halved
    # also confirm ebits>=5 still stores int8
    g = torch.randn(8, 16) * 3; u = torch.randn(8, 16) * 3; d = torch.randn(16, 8) * 3
    mw8, _ = make_merged(g, u, d, 8)
    print(f"  ebits=8: dtype={mw8.dtype} (expected int8), bytes={mw8.numel()}")
    ok = ok and (mw8.dtype == torch.int8)
    # mixed layout: gate/up int4 (gs 8 here), down int8 per row -- the bytes must be
    # exactly the int4 packing of gate|up followed by the int8 rows of down, the size
    # 2*inter*hidden, and the scales [gate ng | up ng | down per row]
    inter, H, gsz = 8, 16, 8
    g = torch.randn(inter, H) * 3; u = torch.randn(inter, H) * 3; d = torch.randn(H, inter) * 3
    mwm, qsm = make_merged(g, u, d, 4, gs=gsz, down_bits=8, down_gs=0)
    gq, _ = quantize_row_grouped(g, 4, gsz); uq, _ = quantize_row_grouped(u, 4, gsz)
    dq, dsc = quantize_row(d, 8)
    gu_ref = pack_int4(torch.cat([gq.flatten(), uq.flatten()]).contiguous())
    n_gu = gu_ref.numel()
    same_gu = bool(torch.equal(mwm[:n_gu], gu_ref))
    same_d = bool(torch.equal(mwm[n_gu:].view(torch.int8), dq.flatten().to(torch.int8)))
    size_ok = mwm.numel() == 2 * inter * H
    ng = H // gsz
    scales_ok = qsm.numel() == 2 * inter * ng + H and bool(torch.allclose(qsm[-H:], dsc.float()))
    print(f"  mixed int4gs{gsz}+down int8: dtype={mwm.dtype} bytes={mwm.numel()} (expected {2*inter*H}) "
          f"gate/up={same_gu} down={same_d} scales={scales_ok}")
    ok = ok and mwm.dtype == torch.uint8 and same_gu and same_d and size_ok and scales_ok
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# Tensor-name contract: classify() places every name or raises. Kept torch-free
# in its own module so the test that pins it to the real checkpoint indexes
# runs wherever the registry tests run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen36_tensor_kinds import classify, resolve_prefix, skip_reason, UnknownTensor  # noqa: E402


MTP_SCHEMA = "colibri.qwen-mtp/v1"
MTP_NATIVE_PRECISION = "qwen36:DBITS=8;IDOT=1;kernel=cpu;tap=post-final-norm-mtp/v1"
MTP_MAX_BYTES = 512 * 1024 * 1024


def _mtp_regular(path):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"MTP needs a regular non-symlink file: {path.name}")
    return path


def _mtp_json(path, limit=1024 * 1024):
    _mtp_regular(path)
    if path.stat().st_size > limit:
        raise ValueError(f"MTP JSON exceeds its byte bound: {path.name}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate MTP JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite MTP JSON constant: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("MTP JSON must be an object")
    return value


def _mtp_int(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"unsupported MTP {name}")
    return value


def _mtp_geometry(config):
    text = config.get("text_config", config)
    if not isinstance(text, dict) or text.get("model_type") not in ("qwen3_5_text", "qwen3_6_text", "qwen3_8_text"):
        raise ValueError("MTP export supports Qwen3.5/3.6/3.8 dense text geometry only")
    if text.get("num_experts", 0) or text.get("num_local_experts", 0):
        raise ValueError("MoE MTP geometry is unsupported")
    if type(text.get("mtp_num_hidden_layers")) is not int or text["mtp_num_hidden_layers"] != 1:
        raise ValueError("MTP export requires exactly one pretrained cell")
    if text.get("mtp_use_dedicated_embeddings", False) is not False:
        raise ValueError("dedicated MTP embeddings are unsupported")
    if text.get("attention_bias", False) is not False or text.get("attn_output_gate", True) is not True:
        raise ValueError("MTP export requires bias-free gated attention")
    geometry = {key: _mtp_int(text.get(key), key, maximum) for key, maximum in (
        ("hidden_size", 32768), ("vocab_size", 1048576), ("intermediate_size", 131072),
        ("num_attention_heads", 256), ("num_key_value_heads", 256), ("head_dim", 1024),
        ("num_hidden_layers", 256), ("max_position_embeddings", 1048576))}
    if geometry["num_attention_heads"] % geometry["num_key_value_heads"]:
        raise ValueError("MTP attention heads must divide evenly into KV heads")
    rope = text.get("rope_parameters", {})
    if not isinstance(rope, dict) or rope.get("rope_type", "default") != "default":
        raise ValueError("MTP rope scaling is unsupported")
    eps = text.get("rms_norm_eps", 1e-6)
    theta = rope.get("rope_theta", text.get("rope_theta", 10000000.0))
    partial = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    for value, name, low, high in ((eps, "rms_norm_eps", 0, 1),
                                 (theta, "rope_theta", 1, 1e12),
                                 (partial, "partial_rotary_factor", 0, 1)):
        if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
            raise ValueError(f"unsupported MTP {name}")
    rotary = geometry["head_dim"] * partial
    if rotary != int(rotary) or int(rotary) % 2 or rotary < 2:
        raise ValueError("unsupported MTP rotary dimension")
    sections = rope.get("mrope_section", text.get("mrope_section", [16, 16, 16]))
    if not isinstance(sections, list) or len(sections) != 3 or any(type(n) is not int or n < 0 or n > 1024 for n in sections):
        raise ValueError("unsupported MTP mrope sections")
    if sum(sections) != int(rotary) // 2 or type(rope.get("mrope_interleaved", True)) is not bool:
        raise ValueError("MTP mrope sections do not cover the rotary dimension")
    geometry.update(rms_norm_eps=eps, rope_theta=theta, partial_rotary_factor=partial,
                    rotary_dim=int(rotary), mrope_section=sections, mtp_num_hidden_layers=1,
                    layer_types=["full_attention"], attn_output_gate=True,
                    rope_parameters={"rope_type": "default", "rope_theta": theta,
                                     "partial_rotary_factor": partial, "mrope_section": sections,
                                     "mrope_interleaved": rope.get("mrope_interleaved", True)})
    return text, geometry


def _mtp_shapes(g):
    h, f, hd = g["hidden_size"], g["intermediate_size"], g["head_dim"]
    q, kv = g["num_attention_heads"] * hd, g["num_key_value_heads"] * hd
    shapes = {"mtp.fc.weight": [h, 2 * h]}
    for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        shapes[f"mtp.{name}.weight"] = [h]
    stem = "mtp.layers.0."
    for name in ("input_layernorm", "post_attention_layernorm"):
        shapes[stem + name + ".weight"] = [h]
    for name, shape in (("q_proj", [2 * q, h]), ("k_proj", [kv, h]), ("v_proj", [kv, h]),
                        ("o_proj", [h, q]), ("q_norm", [hd]), ("k_norm", [hd])):
        shapes[stem + "self_attn." + name + ".weight"] = shape
    for name, shape in (("gate_proj", [f, h]), ("up_proj", [f, h]), ("down_proj", [h, f])):
        shapes[stem + "mlp." + name + ".weight"] = shape
    if sum(math.prod(shape) * 4 for shape in shapes.values()) > MTP_MAX_BYTES:
        raise ValueError("MTP tensor geometry exceeds 512 MiB")
    return shapes


def _mtp_fingerprint(files, tag, precision=None):
    """Streaming byte identities; CSNH tag/precision framing matches the C loader."""
    digest = hashlib.sha256(tag)
    if precision is not None:
        encoded = precision.encode("utf-8", "strict")
        digest.update(struct.pack("<Q", len(encoded))); digest.update(encoded)
    paths = sorted(files, key=lambda path: path.name.encode("utf-8", "strict"))
    if not paths or len(paths) > 256:
        raise ValueError("MTP identity file count exceeds bounds")
    digest.update(struct.pack("<Q", len(paths)))
    records = []
    for path in paths:
        _mtp_regular(path); before = path.stat(); size = before.st_size
        label = path.name.encode("utf-8", "strict")
        if not label or len(label) > 4096 or size < 1 or size > 64 * 1024**3:
            raise ValueError("MTP identity file exceeds bounds")
        digest.update(struct.pack("<Q", len(label))); digest.update(label)
        digest.update(struct.pack("<Q", size)); file_hash = hashlib.sha256(); read = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk); file_hash.update(chunk); read += len(chunk)
        after = path.stat()
        if read != size or (after.st_size, after.st_mtime_ns) != (size, before.st_mtime_ns):
            raise ValueError("MTP identity file changed while hashing")
        records.append({"file": path.name, "bytes": size, "sha256": file_hash.hexdigest()})
    return digest.hexdigest(), records


def export_mtp_sidecar(source_dir, native_dir, output_dir, native_precision, source_revision=None):
    """Export authentic dense MTP tensors only; source/native are read-only.

    Uniform F32 is an exact expansion of F16/BF16 values. Raw zero-centered
    norms are retained: the runtime applies (1 + weight), including q/k norms.
    A fresh external directory is required, since adding a safetensors file to
    the target itself would change its existing native model fingerprint.
    """
    if native_precision != MTP_NATIVE_PRECISION:
        raise ValueError("MTP native precision/tap is not qualified")
    import re
    if source_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("MTP source revision must be an exact lowercase commit SHA")
    source, native, output = map(Path, (source_dir, native_dir, output_dir))
    for root in (source, native):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("MTP source/native must be existing non-symlink directories")
    if output.is_symlink():
        raise ValueError("MTP output aliases an existing symlink")
    source, native, output = source.resolve(), native.resolve(), output.resolve()
    if output == source or source in output.parents or output == native or native in output.parents:
        raise ValueError("MTP output must be outside source and native target directories")
    if output.exists():
        raise ValueError("MTP output directory must be new; overwrite refused")
    config_path = source / "config.json"
    config = _mtp_json(config_path); text, geometry = _mtp_geometry(config)
    shapes = _mtp_shapes(geometry)
    native_config = _mtp_json(native / "config.json")
    original = _mtp_json(native / "config.hf.json")
    meta = _mtp_json(native / "qwen36_meta.json")
    _mtp_geometry(native_config)
    if original != config or native_config != text:
        raise ValueError("MTP source checkpoint config differs from native target")
    expected_meta = {"hidden": geometry["hidden_size"], "n_layers": geometry["num_hidden_layers"],
        "n_active": geometry["num_hidden_layers"], "num_experts": 0, "shared_inter": geometry["intermediate_size"],
        "rms_eps": geometry["rms_norm_eps"], "q_heads": geometry["num_attention_heads"],
        "kv_heads": geometry["num_key_value_heads"], "q_head_dim": 2 * geometry["head_dim"],
        "k_head_dim": geometry["head_dim"], "v_head_dim": geometry["head_dim"],
        "o_in": geometry["num_attention_heads"] * geometry["head_dim"], "head_dim": geometry["head_dim"],
        "rope_theta": geometry["rope_theta"], "partial_rotary_factor": geometry["partial_rotary_factor"]}
    if any(meta.get(key) != value or type(meta.get(key)) is bool for key, value in expected_meta.items()) or \
            meta.get("zero_centered_norms", True) is not True:
        raise ValueError("MTP native geometry/norm convention differs from source")

    index_path = source / "model.safetensors.index.json"
    wm = _mtp_json(index_path, 16 * 1024 * 1024).get("weight_map") if index_path.exists() else None
    shards = {}
    if wm is not None:
        if not isinstance(wm, dict) or not wm or len(wm) > 65536:
            raise ValueError("invalid MTP source weight map")
        for key, name in wm.items():
            if not isinstance(key, str) or len(key) > 512 or not isinstance(name, str) or not name or \
                    Path(name).name != name or any(c in name for c in ("/", "\\", "\0")) or not name.endswith(".safetensors"):
                raise ValueError("unsafe MTP source shard/tensor name")
            shards[name] = source / name
    else:
        wm = {}
        for path in sorted(source.glob("*.safetensors")):
            if path.name != "joint_head.safetensors": shards[path.name] = path
    if not shards or len(shards) > 128:
        raise ValueError("MTP source shard count exceeds bounds")
    descriptions = {}
    for name, path in shards.items():
        _mtp_regular(path)
        with safe_open_pt(str(path), framework="pt") as reader:
            for key in reader.keys():
                if key in descriptions:
                    raise ValueError("duplicate source tensor across MTP shards")
                if index_path.exists() and wm.get(key) != name:
                    raise ValueError("MTP source index/shard mismatch")
                wm[key] = name
                tensor = reader.get_slice(key)
                descriptions[key] = (list(tensor.get_shape()), tensor.get_dtype())
    if set(wm) != set(descriptions):
        raise ValueError("MTP source index has missing tensors")
    prefix = resolve_prefix(wm)
    for key in wm:
        try: classify(key, prefix)
        except UnknownTensor as error: raise ValueError(f"unknown source tensor for MTP export: {key}") from error
    actual = {name for name in descriptions if name.startswith("mtp.")}
    if actual != set(shapes):
        raise ValueError(f"unknown/incomplete MTP tensor set; missing={sorted(set(shapes)-actual)}, extra={sorted(actual-set(shapes))}")
    for name, shape in shapes.items():
        if descriptions[name][0] != shape or descriptions[name][1] not in ("BF16", "F16", "F32"):
            raise ValueError(f"MTP tensor shape/dtype mismatch: {name}")
    for name, shape in ((prefix + "embed_tokens.weight", [geometry["vocab_size"], geometry["hidden_size"]]),
                         (prefix + "norm.weight", [geometry["hidden_size"]])):
        if descriptions.get(name, (None,))[0] != shape:
            raise ValueError(f"MTP shared target tensor shape mismatch: {name}")
    globals_path = _mtp_regular(native / "model-globals.safetensors")
    with safe_open_pt(str(globals_path), framework="pt") as reader:
        for name, shape in (("model.embed_tokens.weight", [geometry["vocab_size"], geometry["hidden_size"]]),
                            ("lm_head.weight", [geometry["vocab_size"], geometry["hidden_size"]]),
                            ("model.norm.weight", [geometry["hidden_size"]])):
            if name not in reader.keys() or list(reader.get_slice(name).get_shape()) != shape:
                raise ValueError(f"MTP native shared tensor shape mismatch: {name}")

    source_files = [config_path, _mtp_regular(source / "tokenizer.json"), *shards.values()]
    if index_path.exists(): source_files.append(index_path)
    source_fingerprint, source_records = _mtp_fingerprint(source_files, b"colibri.qwen-mtp-source/v1\0")
    native_files = [path for path in native.iterdir() if path.name in ("config.json", "qwen36_meta.json", "tokenizer.json") or
                    (len(path.name) > 12 and path.name.endswith(".safetensors"))]
    native_fingerprint, native_records = _mtp_fingerprint(native_files, b"CSNH-model-v1\0", native_precision)
    sr = {item["file"]: item for item in source_records}; nr = {item["file"]: item for item in native_records}
    hf_hash, hf_records = _mtp_fingerprint([native / "config.hf.json"], b"mtp-original-config/v1\0")
    del hf_hash
    if sr["config.json"]["sha256"] != hf_records[0]["sha256"] or sr["tokenizer.json"]["sha256"] != nr["tokenizer.json"]["sha256"]:
        raise ValueError("MTP source config/tokenizer bytes differ from native target")
    tensors = {}
    for shard, path in shards.items():
        names = [name for name in shapes if wm[name] == shard]
        if not names: continue
        with safe_open_pt(str(path), framework="pt") as reader:
            for name in names:
                tensor = reader.get_tensor(name).float().contiguous().clone()
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"nonfinite MTP tensor: {name}")
                tensors[name] = tensor
    from safetensors.torch import save as serialize_tensors
    payload = serialize_tensors(tensors, metadata={"schema": MTP_SCHEMA, "norm_convention": "qwen_rmsnorm_plus_one"})
    if len(payload) > MTP_MAX_BYTES + 1024 * 1024:
        raise ValueError("MTP sidecar exceeds byte limit")
    metadata = {"schema": MTP_SCHEMA, "tensor_dtype": "F32", "norm_convention": "qwen_rmsnorm_plus_one",
        "fc_order": ["embedding", "target_hidden"], "target_hidden_tap": "post_final_norm",
        "shared_embedding": True, "shared_lm_head": True, "geometry": geometry,
        "source": {"checkpoint_fingerprint": source_fingerprint, "config_sha256": sr["config.json"]["sha256"],
                   "tokenizer_sha256": sr["tokenizer.json"]["sha256"], "revision": source_revision, "files": source_records},
        "native": {"fingerprint": native_fingerprint, "runtime_precision": native_precision,
                   "original_config_sha256": hf_records[0]["sha256"], "files": native_records},
        "sidecar": {"file": "model-mtp.safetensors", "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
        "tensors": {name: {"shape": shapes[name], "dtype": "F32", "source_dtype": descriptions[name][1]} for name in sorted(shapes)}}
    document = json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if len(document) > 65536:
        raise ValueError("MTP metadata exceeds 64 KiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # exclusive directory creation, after every validation
    created = []
    try:
        for name, content in (("model-mtp.safetensors", payload), ("qwen36_mtp.json", document)):
            path = output / name
            with path.open("xb") as stream:
                created.append(path); stream.write(content); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        for path in reversed(created): path.unlink()
        try: output.rmdir()
        except OSError: pass
        raise
    return metadata


def main():
    ap = argparse.ArgumentParser(description="Convert Qwen3.6 MoE HF checkpoint -> colibri container")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--repo", help="HuggingFace repo ID (will be downloaded)")
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=False, help="Output container directory")
    ap.add_argument("--ebits", type=int, default=4, help="Expert quant bits (2..8, default 4)")
    ap.add_argument("--down-bits", type=int, default=0,
                    help="bits for down_proj only (5..8; needs --ebits <= 4): the mixed layout, "
                         "int4 gate/up + int8 down in one slab. 0 = same as --ebits")
    ap.add_argument("--down-gs", type=int, default=0,
                    help="scale group size along down_proj's input for --down-bits (0 = per row)")
    ap.add_argument("--gs", type=int, default=0,
                    help="Group size for expert scales (e.g. 64). 0 = per-row (default). "
                         "Group-scaled containers need engine support (expert_gs in meta).")
    ap.add_argument("--upload-repo", help="Push the finished container to this HF repo ID")
    ap.add_argument("--hf-token", help="HF token (defaults to HF_TOKEN env / cached login)")
    ap.add_argument("--low-disk", action="store_true",
                    help="Download source shards one at a time, delete each after its layers are "
                         "converted (only the generated container is kept).")
    ap.add_argument("--stream-upload", action="store_true",
                    help="Upload each generated shard to --upload-repo immediately, then delete it "
                         "locally. Keeps local disk at ~one shard.")
    ap.add_argument("--no-readme", action="store_true", help="Skip README.md generation")
    ap.add_argument("--mtp-only", action="store_true", help="Export pretrained dense MTP to a NEW external sidecar directory; native target stays unchanged")
    ap.add_argument("--native-target", help="Existing native container to bind with --mtp-only")
    ap.add_argument("--native-precision", help="Exact qualified target precision/tap descriptor for --mtp-only")
    ap.add_argument("--source-revision", help="Optional exact source checkpoint commit SHA for --mtp-only")
    ap.add_argument("--selftest", action="store_true",
                    help="Validate the int4 pack/unpack round-trip on synthetic weights, then exit "
                         "(no model conversion). Checks that packing recovers the exact quantized "
                         "integers and that int4 halves the storage vs int8.")
    args = ap.parse_args()

    if args.mtp_only:
        if not args.model or args.repo or not args.out or not args.native_target or not args.native_precision or \
                args.selftest or args.upload_repo or args.stream_upload or args.low_disk:
            ap.error("--mtp-only requires local --model, --native-target, --native-precision and NEW --out; upload/conversion modes are incompatible")
        try:
            metadata = export_mtp_sidecar(args.model, args.native_target, args.out, args.native_precision, args.source_revision)
        except (ValueError, OSError, KeyError) as error:
            sys.exit(f"MTP export refused: {error}")
        print(f"MTP sidecar -> {args.out}; {len(metadata['tensors'])} authentic tensors, native fingerprint {metadata['native']['fingerprint']}")
        return
    if args.native_target or args.native_precision or args.source_revision:
        ap.error("--native-target/--native-precision/--source-revision require --mtp-only")
    if args.selftest:
        return _selftest()
    if not (args.repo or args.model):
        sys.exit("error: need --repo or --model (or --selftest)")
    if not args.out:
        sys.exit("error: --out is required")

    if not 2 <= args.ebits <= 8:
        sys.exit(f"--ebits must be 2..8 (got {args.ebits})")
    if args.down_bits and args.down_bits != args.ebits:
        if not (args.ebits <= 4 and 5 <= args.down_bits <= 8):
            sys.exit(f"--down-bits {args.down_bits} needs --ebits <= 4 and 5..8 (got --ebits {args.ebits})")
        if args.down_gs < 0:
            sys.exit("--down-gs must be >= 0")

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.repo:
        from huggingface_hub import hf_hub_download, HfApi
        api = HfApi(token=token)
    else:
        api = None

    # ---- locate source ----
    if args.repo:
        # download config + index only first (cheap) to learn structure
        cfg_path = hf_hub_download(args.repo, "config.json", token=token)
        try:
            idx_path = hf_hub_download(args.repo, "model.safetensors.index.json", token=token)
        except Exception:
            idx_path = None
        src_dir = None
    else:
        src_dir = Path(args.model)
        if not (src_dir / "config.json").is_file():
            sys.exit(f"config.json missing in {src_dir}")
        cfg_path = str(src_dir / "config.json")
        idx_path = str(src_dir / "model.safetensors.index.json") if (src_dir / "model.safetensors.index.json").is_file() else None

    cfg_full = json.load(open(cfg_path, encoding="utf-8"))
    # VL checkpoints nest dims under text_config
    mcfg = cfg_full.get("text_config", cfg_full)
    # transformers 5 writes qwen3_moe's expert count as num_local_experts; the released
    # checkpoints (written by 4.x) say num_experts. Read either.
    if "num_experts" not in mcfg and "num_local_experts" in mcfg:
        mcfg = dict(mcfg, num_experts=mcfg["num_local_experts"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    # The engine reads FLAT keys from config.json, but VL checkpoints nest them
    # under text_config -> write the flattened text config (the original goes to
    # config.hf.json for reference). Fixes "missing hidden_size" on VL models.
    with open(out / "config.json", "w", encoding="utf-8") as cf:
        json.dump(mcfg, cf, indent=1)
    shutil.copy2(cfg_path, out / "config.hf.json")
    print(f"config -> {out / 'config.json'} (flat; model_type={cfg_full.get('model_type')})")
    # The engine needs tokenizer.json next to the shards (TOK= override exists,
    # but a container should run out of the box).
    try:
        if args.repo:
            tok_path = hf_hub_download(args.repo, "tokenizer.json", token=token)
        else:
            tok_path = str(src_dir / "tokenizer.json")
        if Path(tok_path).is_file():
            shutil.copy2(tok_path, out / "tokenizer.json")
            print(f"tokenizer -> {out / 'tokenizer.json'}")
        else:
            print("WARNING: tokenizer.json not found; the engine will need TOK=<path>")
    except Exception as e:
        print(f"WARNING: could not fetch tokenizer.json ({e}); the engine will need TOK=<path>")
    # The chat template travels with the container: the gateway renders what the
    # checkpoint was trained on, and a Qwen3.8 template on this engine (Qwen3.8-27B,
    # #1757) is recognised from this file.
    for extra in ("chat_template.jinja", "tokenizer_config.json", "generation_config.json",
                  "preprocessor_config.json"):
        try:
            extra_path = (hf_hub_download(args.repo, extra, token=token) if args.repo
                          else str(src_dir / extra))
        except Exception:
            continue
        if Path(extra_path).is_file():
            shutil.copy2(extra_path, out / extra)
            print(f"{extra} -> {out / extra}")
    # A decision head over the backbone (Cloudflare's Clef: docs/clef.md) is two
    # files next to the shards, outside the index. They are copied as they are:
    # the engine reads the head from them, and coli recognises the checkpoint by
    # them. Both or neither: half a head would load as a chat model that cannot
    # say why it does not decide.
    head_files = ("joint_head_config.json", "joint_head.safetensors")
    if args.repo:
        listed = set(HfApi(token=token).list_repo_files(args.repo))
        head_paths = ([hf_hub_download(args.repo, name, token=token) for name in head_files]
                      if all(name in listed for name in head_files) else [])
    else:
        head_paths = [str(src_dir / name) for name in head_files]
    present = [path for path in head_paths if Path(path).is_file()]
    if len(present) == len(head_files):
        for name, path in zip(head_files, head_paths):
            shutil.copy2(path, out / name)
            print(f"{name} -> {out / name}")
    elif present:
        sys.exit(f"ERROR: {Path(present[0]).name} without its sibling: a decision head needs "
                 f"both {' and '.join(head_files)}")

    # ---- build weight map (key -> shard file) ----
    if idx_path:
        wm = json.load(open(idx_path, encoding="utf-8"))["weight_map"]
    else:
        # no index: enumerate local safetensors
        if args.repo:
            sys.exit("remote repo without index.json is unsupported; download the repo first")
        wm = {}
        for sh in sorted(src_dir.glob("*.safetensors")):
            if sh.name in head_files:      # the decision head, copied above
                continue
            with safe_open_np(str(sh), framework="np") as f:
                for k in f.keys():
                    wm[k] = sh.name

    # ---- classify every tensor name before touching a shard ----
    # 5 KB of index settles whether the next hundreds of GB make sense: an
    # unknown name stops here, a skipped subtree is counted and named.
    import re
    prefix = resolve_prefix(wm.keys())
    layer_map = {}      # layer index -> [keys]
    global_map = {}     # kind -> key
    vision_map = {}     # name after "visual." -> key
    skipped = {}        # group -> count
    unknown = []
    for k in wm:
        try:
            placed = classify(k, prefix)
        except UnknownTensor:
            unknown.append(k)
            continue
        if placed[0] == "layer":
            layer_map.setdefault(placed[1], []).append(k)
        elif placed[0] == "global":
            global_map[placed[1]] = k
        elif placed[0] == "vision":
            vision_map[placed[1]] = k
        else:
            skipped[placed[1]] = skipped.get(placed[1], 0) + 1
    if unknown:
        shown = "\n  ".join(unknown[:12])
        more = f"\n  ... and {len(unknown) - 12} more" if len(unknown) > 12 else ""
        sys.exit(f"ERROR: {len(unknown)} tensor name(s) this converter cannot place "
                 f"(tools/qwen36_tensor_kinds.py):\n  {shown}{more}\n"
                 "Refusing to convert: a container missing or blindly carrying a tensor "
                 "loads and answers wrongly. Extend the contract if the name is legitimate.")
    for group, count in sorted(skipped.items()):
        declared = ""
        if group == "mtp":
            declared = f", config mtp_num_hidden_layers={mcfg.get('mtp_num_hidden_layers')!r}"
        print(f"skipping {count} `{group}.*` tensor(s): {skip_reason(group)}{declared}")
    if mcfg.get("mtp_num_hidden_layers") and "mtp" not in skipped:
        print(f"note: config declares mtp_num_hidden_layers={mcfg['mtp_num_hidden_layers']} "
              "but the checkpoint carries no mtp.* tensors")
    layer_ids = set(layer_map)
    n_layers_weight = (max(layer_ids) + 1) if layer_ids else mcfg.get("num_hidden_layers", 0)
    print(f"tensor kinds: {sum(len(v) for v in layer_map.values())} in {len(layer_ids)} layers, "
          f"{len(global_map)} globals, prefix {prefix!r}")

    # Qwen3 MoE (qwen3_moe, e.g. Qwen3-Coder-30B-A3B): every layer is plain attention,
    # full rotary, ordinary RMSNorm weights and no shared expert. Qwen3.5/3.6 differ on
    # each of those, so the defaults below branch on it.
    plain_qwen3 = cfg_full.get("model_type") == "qwen3_moe"
    layer_types = mcfg.get("layer_types")
    if layer_types is None:
        n = mcfg.get("num_hidden_layers", n_layers_weight)
        if plain_qwen3:
            layer_types = ["full_attention"] * n
        else:
            # infer: every 4th layer is full_attention (Qwen3.5/3.6 convention)
            layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(n)]
    full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    all_idx = list(range(len(layer_types)))
    print(f"layers: {len(layer_types)} total, {len(full_idx)} full_attention (Phase 2: all {len(all_idx)} stored)")

    # ---- shard cache / eviction ----
    tmp = out / ".src_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    local_shards = {}     # shard name -> local path
    shard_handles = {}    # shard name -> safe_open handle
    shard_layers = {}     # shard name -> set of layer ids needing it (-1 = globals)

    def layer_keys(i):
        return layer_map.get(i, [])
    for i in all_idx:
        for k in layer_keys(i):
            shard_layers.setdefault(wm[k], set()).add(i)
    missing_layers = [i for i in all_idx if i not in layer_map]
    if missing_layers:
        sys.exit(f"ERROR: config declares {len(all_idx)} layers but the checkpoint has no "
                 f"tensors for layer(s) {missing_layers[:8]}")
    globals_keys = [global_map[kind] for kind in ("embed_tokens.weight", "lm_head.weight",
                                                  "norm.weight") if kind in global_map]
    tied_head = ("lm_head.weight" not in global_map and
                 (mcfg.get("tie_word_embeddings") is True or cfg_full.get("tie_word_embeddings") is True))
    if len(globals_keys) != 3 and not (tied_head and
            "embed_tokens.weight" in global_map and "norm.weight" in global_map):
        sys.exit(f"ERROR: expected embed_tokens, lm_head and final norm, found {sorted(global_map)}")
    for k in globals_keys:
        shard_layers.setdefault(wm[k], set()).add(-1)

    def ensure_shard(name):
        if name in local_shards:
            return local_shards[name]
        if args.repo:
            p = hf_hub_download(args.repo, name, token=token, local_dir=str(tmp),
                                local_dir_use_symlinks=False)
        else:
            p = str(src_dir / name)
        local_shards[name] = p
        return p

    def get_tensor(name):
        shard = wm[name]
        ensure_shard(shard)
        if shard not in shard_handles:
            # torch backend returns a native torch tensor (bf16 works natively);
            # numpy backend returns a bf16 numpy array that torch.from_numpy rejects.
            shard_handles[shard] = safe_open_pt(local_shards[shard], framework="pt")
        return shard_handles[shard].get_tensor(name).contiguous()

    def evict_layer(i):
        for shard, lays in list(shard_layers.items()):
            lays.discard(i)
            if not lays:
                if args.repo:   # only delete files we downloaded; for local keep them on disk
                    h = shard_handles.pop(shard, None)
                    try:
                        if h is not None:
                            h.__exit__(None, None, None)
                    except Exception:
                        pass
                    p = local_shards.pop(shard, None)
                    if p is not None:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                # For local source we keep the handle open (the file stays on disk and
                # the post-loop meta head-dim derivation re-reads it). Only the remote
                # path pops/closes, since it deletes the downloaded shard.
                shard_layers.pop(shard, None)

    stream_api = None
    if args.stream_upload or args.upload_repo:
        from huggingface_hub import HfApi
        stream_api = HfApi(token=token)
        if args.upload_repo:
            stream_api.create_repo(args.upload_repo, repo_type="model", exist_ok=True, private=False)

    def upload_local(path: Path):
        if stream_api and args.upload_repo:
            stream_api.upload_file(repo_id=args.upload_repo, repo_type="model",
                                   path_in_repo=path.name, path_or_fileobj=str(path))
            print(f"  streamed -> {args.upload_repo}/{path.name}")

    # ---- resume: discover already-uploaded shards so a restart skips them ----
    done_files = set()
    if args.upload_repo and stream_api is not None:
        try:
            done_files = set(stream_api.list_repo_files(args.upload_repo, repo_type="model"))
            print(f"[resume] {len(done_files)} file(s) already on {args.upload_repo}; "
                  f"will skip finished layers.")
        except Exception as e:
            done_files = set()
            print(f"[resume] could not list repo ({e}); running full conversion.")

    # ---- globals shard (embed / lm_head / final norm) as f16 ----
    if "model-globals.safetensors" in done_files:
        print("[globals] already on HF, skip")
    else:
        g_out = {}
        for k in globals_keys:
            arr = get_tensor(k).half()
            newk = k.replace("model.language_model.", "model.")
            g_out[newk] = arr
        if tied_head:
            # Separate storage is required by safetensors; tying is an explicit
            # source-model contract, never a guess for an incomplete checkpoint.
            embed_key = global_map["embed_tokens.weight"].replace("model.language_model.", "model.")
            g_out["lm_head.weight"] = g_out[embed_key].clone()
            print("[globals] materialized explicitly tied lm_head from embeddings")
        gpath = out / "model-globals.safetensors"
        save_file(g_out, str(gpath))
        print(f"[globals] {gpath.name} ({len(g_out)} tensors)")
        if args.stream_upload:
            upload_local(gpath)
            gpath.unlink()

    # ---- vision tower as f16, under model.visual.* (#1757) ----
    # The same ViT in Qwen3.5/3.6/3.8; the engine runs it through qwen38_vision.h.
    vcfg = cfg_full.get("vision_config") or {}
    if vision_map and not vcfg:
        print(f"note: {len(vision_map)} visual.* tensor(s) but no vision_config; the tower is left out")
    if vision_map and vcfg:
        if "model-vision.safetensors" in done_files:
            print("[vision] already on HF, skip")
        else:
            v_out = {f"model.visual.{name}": get_tensor(k).half() for name, k in vision_map.items()}
            vpath = out / "model-vision.safetensors"
            save_file(v_out, str(vpath))
            print(f"[vision] {vpath.name} ({len(v_out)} tensors)")
            if args.stream_upload:
                upload_local(vpath)
                vpath.unlink()

    # ---- per layer: Gated-Attention (full_attention) AND Gated DeltaNet (linear_attention) ----
    # Every layer also carries its MoE/MLP block. DeltaNet layers export linear_attn.* which
    # the generic f16 copy below handles automatically; the engine implements the recurrence.
    active = 0
    for i in all_idx:
        a = active
        active += 1
        out_name = f"model-{a:05d}.safetensors"
        if out_name in done_files:
            print(f"[layer {i} -> active {a}] {out_name} already on HF, skip")
            evict_layer(i)
            continue
        tens = {}
        lk = layer_keys(i)
        # ---- MoE experts: handle BOTH layouts the source may use ----
        #  (a) FUSED:  model.layers.i.mlp.experts.gate_up_proj [E,2*inter,H]
        #              + model.layers.i.mlp.experts.down_proj    [E,H,inter]
        #  (b) SEPARATE (transformers save_pretrained on the text model, e.g. the tiny fixture;
        #      the real 35B and 2.4T checkpoints both ship the FUSED layout):
        #              model.layers.i.mlp.experts.{e}.gate_proj [inter,H]
        #              model.layers.i.mlp.experts.{e}.up_proj   [inter,H]
        #              model.layers.i.mlp.experts.{e}.down_proj [H,inter]
        sep = {}  # e -> {'gate':Tensor,'up':Tensor,'down':Tensor}
        for k in lk:
            if re.search(r"\.experts\.gate_up_proj(?:\.weight)?$", k):
                gu = get_tensor(k).float()            # [E, 2*inter, H]
                E, twoI, H = gu.shape
                inter = twoI // 2
                gate = gu[:, :inter, :]
                up = gu[:, inter:, :]
                dk = k.replace("gate_up_proj", "down_proj")
                down = get_tensor(dk).float()        # [E, H, inter]
                for e in range(E):
                    mw, qs = make_merged(gate[e], up[e], down[e], args.ebits, gs=args.gs, down_bits=args.down_bits, down_gs=args.down_gs)
                    tens[f"model.layers.{a}.mlp.experts.{e}.merged_weight"] = mw
                    tens[f"model.layers.{a}.mlp.experts.{e}.qs"] = qs
                continue
            ms = re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)(?:\.weight)?$", k)
            if ms:
                e = int(ms.group(1)); proj = ms.group(2)
                sep.setdefault(e, {})[proj] = get_tensor(k).float()
                continue
        for e in sorted(sep):
            d = sep[e]
            if "gate_proj" in d and "up_proj" in d and "down_proj" in d:
                mw, qs = make_merged(d["gate_proj"], d["up_proj"], d["down_proj"], args.ebits, gs=args.gs, down_bits=args.down_bits, down_gs=args.down_gs)
                tens[f"model.layers.{a}.mlp.experts.{e}.merged_weight"] = mw
                tens[f"model.layers.{a}.mlp.experts.{e}.qs"] = qs
            else:
                print(f"[layer {i}] WARN expert {e} missing some proj "
                      f"(have {sorted(d.keys())}) — skipped")
        for k in lk:
            newk = k.replace(f"{prefix}layers.{i}.", f"model.layers.{a}.")
            if "mlp.experts." in newk:
                continue  # already handled above
            # everything else stays f16
            tens[newk] = get_tensor(k).half()
        out_path = out / f"model-{a:05d}.safetensors"
        save_file(tens, str(out_path))
        print(f"[layer {i} -> active {a}] {out_path.name} ({len(tens)} tensors)")
        if args.stream_upload:
            upload_local(out_path)
            out_path.unlink()
        evict_layer(i)

    # ---- derive authoritative head dims from a real full-attention layer's weights ----
    meta = {
        "model_type": cfg_full.get("model_type"),
        "hidden": int(mcfg["hidden_size"]),
        "n_layers": int(mcfg["num_hidden_layers"]),
        "n_active": len(all_idx),
        "layer_types": layer_types,
        # A dense checkpoint of the family (no num_experts: Qwen3.5 / Qwen3.8 27B) routes
        # nothing; the engine loads its MLP as an ungated shared expert of that width.
        "num_experts": int(mcfg.get("num_experts", 0)),
        "topk": int(mcfg.get("num_experts_per_tok", 0)) if mcfg.get("num_experts") else 0,
        "moe_inter": (int(mcfg.get("moe_intermediate_size", mcfg.get("intermediate_size", 0) // 2))
                      if mcfg.get("num_experts") else 0),
        "shared_inter": (0 if plain_qwen3 else
                         int(mcfg.get("shared_expert_intermediate_size", mcfg.get("moe_intermediate_size", 0)))
                         if mcfg.get("num_experts") else int(mcfg["intermediate_size"])),
        "rms_eps": float(mcfg.get("rms_norm_eps", 1e-6)),
        "ebits": args.ebits,
        "scoring_func": mcfg.get("scoring_func", "softmax"),
        "n_group": int(mcfg.get("n_group", 1)),
        "topk_group": int(mcfg.get("topk_group", 1)),
        "norm_topk_prob": bool(mcfg.get("norm_topk_prob", False)),
        "attn_output_gate": bool(mcfg.get("attn_output_gate", False)),
        "rope_theta": float((mcfg.get("rope_parameters") or {}).get(
            "rope_theta", mcfg.get("rope_theta", 10000000.0))),
        "mrope_section": (mcfg.get("rope_parameters") or {}).get("mrope_section", [16, 16, 16]),
        "partial_rotary_factor": float(mcfg.get("partial_rotary_factor", 1.0 if plain_qwen3 else 0.25)),
    }
    if plain_qwen3:
        meta["zero_centered_norms"] = False
    # derive head dims from first full-attention layer weights (authoritative).
    # Read from SOURCE (not local output) so this works even when layer 0 was
    # skipped during a resume and its local shard was already stream-deleted.
    if full_idx:
        fi = full_idx[0]
        def shp_src(nm):
            k = f"{prefix}layers.{fi}.self_attn.{nm}.weight"
            if k in wm:
                return tuple(get_tensor(k).shape)
            return None
        qp = shp_src("q_proj")
        kp = shp_src("k_proj")
        vp = shp_src("v_proj")
        op = shp_src("o_proj")
        qn = shp_src("q_norm")
        n_q = int(mcfg["num_attention_heads"])
        n_kv = int(mcfg["num_key_value_heads"])
        meta["q_heads"] = n_q
        meta["kv_heads"] = n_kv
        if qp is not None:
            meta["q_head_dim"] = qp[0] // n_q
        if kp is not None:
            meta["k_head_dim"] = kp[0] // n_kv
        if vp is not None:
            meta["v_head_dim"] = vp[0] // n_kv
        if op is not None:
            meta["o_in"] = op[1]
        if qn is not None:
            meta["qk_rope_head_dim"] = qn[0]
        meta["expert_gs"] = args.gs   # 0 = per-row scales; >0 = group size along input dim
        if args.down_bits and args.down_bits != args.ebits:
            # mixed layout (docstring): down_proj in its own format, gate/up as ebits/gs
            meta["expert_down_bits"] = args.down_bits
            meta["expert_down_gs"] = args.down_gs
        meta["head_dim"] = meta.get("k_head_dim", meta.get("q_head_dim", 256))
        meta["rope_dim"] = meta.get("qk_rope_head_dim", meta["head_dim"] // 4)
        if plain_qwen3:
            # One position per token, so M-RoPE over sections that cover the whole
            # rotary half is ordinary 1D RoPE; the engine needs the sections to add up.
            half = int(meta["rope_dim"] * meta["partial_rotary_factor"]) // 2
            meta["mrope_section"] = [half - 2 * (half // 3), half // 3, half // 3]
    # ---- DeltaNet (linear_attention) dims. From config (authoritative for dn; unlike the
    #      attention qk split, dn dims are explicit in config and never evicted shards). ----
    meta["dn_vheads"] = int(mcfg.get("linear_num_value_heads", mcfg.get("num_value_heads", 0)))
    meta["dn_kheads"] = int(mcfg.get("linear_num_key_heads", mcfg.get("num_key_heads", 0)))
    meta["dn_kdim"] = int(mcfg.get("linear_key_head_dim", mcfg.get("key_head_dim", 0)))
    meta["dn_vdim"] = int(mcfg.get("linear_value_head_dim", mcfg.get("value_head_dim", 0)))
    meta["dn_convk"] = int(mcfg.get("linear_conv_kernel_dim", mcfg.get("conv_kernel_size", 0)))
    meta["dn_conv_dim"] = meta["dn_kheads"] * meta["dn_kdim"] * 2 + meta["dn_vheads"] * meta["dn_vdim"]
    # close remaining source handles now that we've finished reading for meta
    for h in shard_handles.values():
        try:
            h.__exit__(None, None, None)
        except Exception:
            pass
    if vision_map and vcfg:
        meta["image_token_id"] = int(cfg_full["image_token_id"])
        meta["vision"] = {
            "depth": int(vcfg["depth"]), "hidden": int(vcfg["hidden_size"]),
            "heads": int(vcfg["num_heads"]), "inter": int(vcfg["intermediate_size"]),
            "patch": int(vcfg["patch_size"]), "merge": int(vcfg["spatial_merge_size"]),
            "temporal": int(vcfg["temporal_patch_size"]), "in_ch": int(vcfg["in_channels"]),
            "out_hidden": int(vcfg["out_hidden_size"]),
            "num_pos": int(vcfg["num_position_embeddings"]),
        }
    (out / "qwen36_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] {out / 'qwen36_meta.json'}")

    # ---- README ----
    if not args.no_readme and plain_qwen3:
        lines = ["---", "tags:", "  - colibri", "  - qwen3_moe", "  - moe",
                 "library_name: colibri", "---", "",
                 "colibri container for a Qwen3 MoE checkpoint (qwen3_moe, e.g. Qwen3-Coder-30B-A3B):",
                 f"{len(all_idx)} attention layers, no DeltaNet and no shared expert, routed experts",
                 f"at {args.ebits} bits" + (f" with a scale per {args.gs} inputs." if args.gs else " with a scale per row."),
                 "", "Engine: https://github.com/JustVugg/colibri (c/qwen36.c, docs/qwen36.md)"]
        (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif not args.no_readme:
        lines = ["---", "tags:", "  - colibri", "  - qwen3.6", "  - qwen3.6-35b-a3b", "  - moe",
                 "library_name: colibri", "---", "",
                 "colibri container for Qwen3.6-35B-A3B (Phase 2: all layers, incl. Gated DeltaNet).",
                 "Every layer (Gated-Attention + Gated DeltaNet linear_attention) carries its",
                 "MoE/MLP block. DeltaNet weights live under model.layers.{i}.linear_attn.* and are",
                 "run by the recurrent gated-delta-rule in the colibri `qwen36` engine.",
                 "", "Engine: https://github.com/JustVugg/colibri (c/qwen36.c)"]
        (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- final upload (non-stream path) + cleanup ----
    if args.upload_repo:
        print(f"\nUploading to HF Hub: {args.upload_repo}")
        for f in sorted(out.glob("*.safetensors")) + [out / "config.json", out / "config.hf.json",
                                                       out / "tokenizer.json",
                                                       out / "qwen36_meta.json", out / "README.md"]:
            if f.is_file():
                stream_api.upload_file(repo_id=args.upload_repo, repo_type="model",
                                       path_in_repo=f.name, path_or_fileobj=str(f))
        print(f"Uploaded. Pull on your laptop with:\n"
              f"  python -c \"from huggingface_hub import snapshot_download; "
              f"print(snapshot_download('{args.upload_repo}'))\"")
    # never delete the generated container
    try:
        if tmp.is_dir():
            import shutil as _sh
            _sh.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    print(f"\nDone. Container at: {out}")
    n_linear = sum(kind == "linear_attention" for kind in layer_types)
    print(f"All {len(all_idx)} layers stored ({n_linear} Gated DeltaNet). n_active={len(all_idx)}.")
    print(f"Run (engine):  SNAP={out} ./qwen36 16 {args.ebits} ref_qwen36.json")


if __name__ == "__main__":
    main()

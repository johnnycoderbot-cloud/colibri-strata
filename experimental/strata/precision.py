"""CSQ1 signed INT4/INT8/INT16 storage and trusted quantized-head C export.

Integer weights are dequantized for reference F32 activations/compute. Integer
storage does not establish native integer arithmetic on an accelerator.
"""
import math
import re
import struct

import heads

MAX_VALUES = 16777216
MAX_BLOCK = 4096


def f32(value):
    heads.require(type(value) in (int, float) and abs(value) <= heads.FLOAT32_MAX
                  and math.isfinite(value), "value must be finite F32")
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(values, bits, block_values=64):
    heads.require(type(bits) is int and bits in (4, 8, 16), "bits must be 4,8,16")
    heads.positive_int(block_values, "block_values", MAX_BLOCK)
    heads.require(len(values) <= MAX_VALUES, "too many values")
    src = [f32(v) for v in values]
    blocks = (len(src) + block_values - 1) // block_values
    result = bytearray(struct.pack("<4s5I", b"CSQ1", 1, bits, len(src), block_values, blocks))
    limit = (1 << (bits - 1)) - 1
    for start in range(0, len(src), block_values):
        block = src[start:start + block_values]
        maximum = max(map(abs, block), default=0)
        result.extend(struct.pack("<f", maximum))
        codes = []
        for v in block:
            scaled = (v / maximum) * limit if maximum else 0
            q = math.trunc(scaled + (-0.5 if scaled < 0 else 0.5))
            codes.append(max(-limit, min(limit, q)))
        if bits == 4:
            for i in range(0, len(codes), 2):
                result.append((codes[i] & 15) | ((codes[i + 1] & 15) << 4 if i + 1 < len(codes) else 0))
        elif bits == 8:
            result.extend(q & 255 for q in codes)
        else:
            result.extend(struct.pack("<" + "h" * len(codes), *codes))
    return bytes(result)


def decode(blob):
    heads.require(isinstance(blob, bytes) and len(blob) >= 24, "truncated CSQ1")
    magic, version, bits, count, block_values, blocks = struct.unpack_from("<4s5I", blob)
    heads.require(magic == b"CSQ1" and version == 1 and bits in (4, 8, 16), "invalid CSQ1 header")
    heads.positive_int(block_values, "block_values", MAX_BLOCK)
    heads.require(count <= MAX_VALUES and blocks == (count + block_values - 1) // block_values,
                  "invalid CSQ1 dimensions")
    pos, values = 24, []
    limit = (1 << (bits - 1)) - 1
    for start in range(0, count, block_values):
        n = min(block_values, count - start)
        payload = (n + 1) // 2 if bits == 4 else n * (bits // 8)
        heads.require(pos + 4 + payload <= len(blob), "truncated CSQ1 block")
        maximum = struct.unpack_from("<f", blob, pos)[0]
        scale_bits = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        heads.require(math.isfinite(maximum) and maximum >= 0, "invalid CSQ1 scale")
        heads.require(maximum != 0 or scale_bits == 0, "noncanonical CSQ1 negative zero scale")
        if bits == 4:
            if n % 2:
                heads.require(blob[pos + payload - 1] >> 4 == 0, "noncanonical INT4 padding")
            codes = [(blob[pos + i // 2] >> (4 * (i % 2))) & 15 for i in range(n)]
            codes = [q - 16 if q >= 8 else q for q in codes]
        elif bits == 8:
            codes = [q - 256 if q >= 128 else q for q in blob[pos:pos + n]]
        else:
            codes = list(struct.unpack_from("<" + "h" * n, blob, pos))
        heads.require(all(-limit <= q <= limit and (maximum != 0 or q == 0) for q in codes),
                      "reserved or invalid CSQ1 integer code")
        heads.require(maximum == 0 or any(abs(q) == limit for q in codes),
                      "noncanonical CSQ1 block without an extreme code")
        values.extend(f32(q * (maximum / limit)) for q in codes)
        pos += payload
    heads.require(pos == len(blob), "CSQ1 trailing bytes")
    return {"bits": bits, "count": count, "block_values": block_values,
            "block_count": blocks, "encoded_bytes": len(blob)}, values


def quantized_weights(pack, bits, block_values):
    heads.validate_pack(pack)
    return [{key: encode(head[key], bits, block_values) for key in ("down", "up")}
            for head in pack["heads"]]


def export_c(pack, bits, block_values=64, prefix="strata_trained"):
    meta = heads.validate_pack(pack)
    heads.require(bool(re.fullmatch(r"[a-z][a-z0-9_]*", prefix)), "invalid C symbol prefix")
    weights = quantized_weights(pack, bits, block_values)
    lines = ['/* Generated quantized-storage heads; reference F32 compute. */',
             '#include "strata_quant_heads.h"']
    for key in ("family", "checkpoint_revision", "tokenizer_sha256", "hidden_tap", "quantization"):
        lines.append(f"static const char {prefix}_{key}[] = {heads.c_string(meta[key])};")
    for i, head in enumerate(pack["heads"]):
        for key in ("down", "up"):
            blob = weights[i][key]
            lines.append(f"static const uint8_t {prefix}_{i}_{key}[] = {{")
            lines.extend("    " + ",".join(str(b) for b in blob[j:j + 24]) + "," for j in range(0, len(blob), 24))
            lines.append("};")
        literals = []
        for v in head["bias"]:
            text = format(f32(v), ".9g")
            if "." not in text and "e" not in text.lower():
                text += ".0"
            literals.append(text + "f")
        lines.append(f"static const float {prefix}_{i}_bias[] = {{" + ",".join(literals) + "};")
    lines.append(f"static const strata_quant_heads_head {prefix}_heads[] = {{")
    for i, head in enumerate(pack["heads"]):
        lines.append("    {" + f".future_offset={head['future_offset']}, .rank={head['rank']}, "
                     f".down={prefix}_{i}_down, .down_bytes={len(weights[i]['down'])}, "
                     f".up={prefix}_{i}_up, .up_bytes={len(weights[i]['up'])}, "
                     f".bias={prefix}_{i}_bias, .bias_count={meta['vocab_size']}" + "},")
    lines.append("};")
    lines.append(f"static const strata_quant_heads_bank {prefix}_bank = {{"
                 f".hidden_dim={meta['hidden_size']}, .vocab_size={meta['vocab_size']}, "
                 f".head_count={len(pack['heads'])}, .heads={prefix}_heads" + "};")
    return "\n".join(lines) + "\n"


def reference(pack, hidden, bits=None, block_values=64):
    meta = heads.validate_pack(pack)
    heads.require(len(hidden) == meta["hidden_size"], "hidden width mismatch")
    hidden = [f32(v) for v in hidden]
    result = []
    for head in pack["heads"]:
        down, up = [f32(v) for v in head["down"]], [f32(v) for v in head["up"]]
        if bits is not None:
            down = decode(encode(down, bits, block_values))[1]
            up = decode(encode(up, bits, block_values))[1]
        rank, width = head["rank"], meta["hidden_size"]
        latent = [f32(sum(down[r * width + d] * hidden[d] for d in range(width))) for r in range(rank)]
        logits = [f32(sum(up[v * rank + r] * latent[r] for r in range(rank)))
                  for v in range(meta["vocab_size"])]
        logits = [f32(v + f32(b)) for v, b in zip(logits, head["bias"])]
        result.append(logits)
    return result

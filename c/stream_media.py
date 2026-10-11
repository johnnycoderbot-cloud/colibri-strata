"""Bounded media codecs; no microphone, camera, model or device access."""
import hashlib
import importlib
import io
import math
import struct
import zlib

SAMPLE_RATE = 16000
DEFAULT_WAV_BYTES = 2 * 1024 * 1024
DEFAULT_PNG_BYTES = 4 * 1024 * 1024
DEFAULT_OUTPUT_PIXELS = (16 * 1024 * 1024) // 3
MAX_OUTPUT_PIXELS = 16 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dependency(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"optional dependency {name!r} is unavailable") from exc


def _wav_parts(blob, max_audio_seconds, max_bytes):
    require(type(max_bytes) is int and max_bytes > 0, "invalid WAV byte limit")
    require(type(max_audio_seconds) in (int, float) and math.isfinite(max_audio_seconds)
            and 0 < max_audio_seconds <= 30, "audio duration limit must be in (0,30]")
    require(isinstance(blob, bytes) and 12 <= len(blob) <= max_bytes, "WAV payload exceeds bounds")
    require(blob[:4] == b"RIFF" and blob[8:12] == b"WAVE" and struct.unpack_from("<I", blob, 4)[0] == len(blob) - 8,
            "invalid or noncanonical RIFF/WAVE framing")
    pos, fmt, data = 12, None, None
    while pos < len(blob):
        require(pos + 8 <= len(blob), "truncated WAV chunk header")
        kind, size = struct.unpack_from("<4sI", blob, pos)
        pos += 8
        require(size <= len(blob) - pos, "truncated WAV chunk")
        chunk = memoryview(blob)[pos:pos + size]
        if kind == b"fmt ":
            require(fmt is None and data is None and size in (16, 18), "unsupported or duplicate WAV format")
            fmt = struct.unpack_from("<HHIIHH", chunk)
            if size == 18:
                require(struct.unpack_from("<H", chunk, 16)[0] == 0, "unsupported WAV format extension")
        elif kind == b"data":
            require(fmt is not None and data is None, "WAV needs one data chunk after its format")
            data = chunk
        pos += size
        if size % 2:
            require(pos < len(blob) and blob[pos] == 0, "missing or noncanonical WAV padding")
            pos += 1
    require(fmt is not None and data is not None, "missing WAV format or samples")
    code, channels, rate, byte_rate, block_align, bits = fmt
    require(channels == 1 and rate == SAMPLE_RATE, "WAV must be mono at 16000 Hz")
    require((code, bits) in ((1, 16), (3, 32)), "WAV must use PCM16 or IEEE float32 samples")
    sample_bytes = bits // 8
    require(block_align == sample_bytes and byte_rate == SAMPLE_RATE * sample_bytes,
            "WAV byte-rate/block-alignment mismatch")
    require(len(data) > 0 and len(data) % sample_bytes == 0, "incomplete or empty WAV samples")
    count = len(data) // sample_bytes
    # Whisper's cancellation streamer is supported only for short-form <30s.
    require(count <= int(max_audio_seconds * SAMPLE_RATE) and count < 30 * SAMPLE_RATE,
            "WAV exceeds short-form duration limit (<30 seconds)")
    if code == 3:
        require(all(math.isfinite(v) and -1 <= v <= 1 for (v,) in struct.iter_unpack("<f", data)),
                "float WAV samples must be finite and normalized to [-1,1]")
    meta = {"sample_rate": SAMPLE_RATE, "channels": 1, "sample_count": count,
            "duration": count / SAMPLE_RATE, "encoding": "pcm16" if code == 1 else "float32"}
    return meta, data


def validate_wav(wav_bytes, max_audio_seconds=30, max_bytes=DEFAULT_WAV_BYTES):
    """Validate independent mono16k short WAV chunks using only the stdlib."""
    return _wav_parts(wav_bytes, max_audio_seconds, max_bytes)[0]


def decode_wav(wav_bytes, max_audio_seconds=30, max_bytes=DEFAULT_WAV_BYTES):
    meta, data = _wav_parts(wav_bytes, max_audio_seconds, max_bytes)
    np = dependency("numpy")
    samples = np.frombuffer(data, dtype="<i2" if meta["encoding"] == "pcm16" else "<f4").astype(np.float32)
    if meta["encoding"] == "pcm16":
        samples *= 1.0 / 32768.0
    return {**meta, "samples": samples}


def encode_wav_pcm16(samples, max_audio_seconds=30):
    """Encode normalized mono samples; callers decide whether capture is allowed."""
    np = dependency("numpy")
    values = np.asarray(samples)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    require(values.ndim == 1 and values.dtype.kind in "iuf" and len(values) > 0,
            "audio samples must be a nonempty mono real-number vector")
    require(bool(np.all(np.isfinite(values))) and bool(np.all(np.abs(values) <= 1)), "audio samples must be finite in [-1,1]")
    require(len(values) <= int(max_audio_seconds * SAMPLE_RATE) and len(values) < 30 * SAMPLE_RATE,
            "audio exceeds short-form duration limit")
    payload = np.clip(np.rint(values.astype(np.float64) * 32768), -32768, 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    result = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    validate_wav(result, max_audio_seconds=max_audio_seconds)
    return result


def _png_header(blob, max_bytes, max_output_pixels, scale):
    require(type(max_bytes) is int and max_bytes > 0 and type(max_output_pixels) is int
            and 1 <= max_output_pixels <= MAX_OUTPUT_PIXELS and type(scale) is int and 1 <= scale <= 16,
            "invalid PNG bounds")
    require(isinstance(blob, bytes) and 45 <= len(blob) <= max_bytes and blob.startswith(PNG_SIGNATURE), "invalid or oversized PNG payload")
    pos, dimensions, saw_idat, saw_iend = 8, None, False, False
    while pos < len(blob):
        require(pos + 12 <= len(blob), "truncated PNG chunk")
        length = struct.unpack_from(">I", blob, pos)[0]
        kind = blob[pos + 4:pos + 8]
        require(all(65 <= c <= 90 or 97 <= c <= 122 for c in kind), "invalid PNG chunk type")
        require(length <= len(blob) - pos - 12, "truncated PNG chunk payload")
        data = blob[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", blob, pos + 8 + length)[0]
        require(zlib.crc32(kind + data) & 0xffffffff == crc, "PNG chunk CRC mismatch")
        if dimensions is None:
            require(kind == b"IHDR" and length == 13, "PNG must start with IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            require(width > 0 and height > 0 and depth == 8 and color in (0, 2, 3, 4, 6)
                    and compression == filtering == 0 and interlace in (0, 1), "unsupported PNG pixel format")
            require(width * height * scale * scale <= max_output_pixels, "PNG exceeds scaled output pixel limit")
            dimensions = (width, height)
        elif kind == b"IHDR":
            raise ValueError("duplicate PNG IHDR")
        if kind == b"IDAT":
            saw_idat = True
        elif kind == b"IEND":
            require(length == 0 and saw_idat and pos + 12 == len(blob), "noncanonical PNG end/trailing data")
            saw_iend = True
        elif kind[:1].isupper():
            require(kind in (b"IHDR", b"PLTE"), "unsupported critical PNG chunk")
        pos += length + 12
    require(dimensions is not None and saw_iend, "missing PNG pixels or end")
    return dimensions


def _load_png(blob, max_bytes, max_output_pixels, scale):
    width, height = _png_header(blob, max_bytes, max_output_pixels, scale)
    Image = dependency("PIL.Image")
    try:
        with Image.open(io.BytesIO(blob)) as image:
            require(image.format == "PNG" and getattr(image, "n_frames", 1) == 1, "only single-frame PNG is supported")
            image.load()
            require(image.size == (width, height), "PNG decoder dimensions disagree with IHDR")
            alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
            pixels = image.convert("RGBA" if alpha else "RGB").copy()
    except (OSError, SyntaxError) as exc:
        raise ValueError("PNG pixels could not be decoded") from exc
    return pixels, {"width": width, "height": height, "channels": 4 if alpha else 3,
                    "output_width": width * scale, "output_height": height * scale,
                    "output_pixels": width * height * scale * scale}


def validate_png(png_bytes, max_output_pixels=DEFAULT_OUTPUT_PIXELS, scale=4, max_bytes=DEFAULT_PNG_BYTES):
    """Validate PNG framing, CRCs, decoded pixels and scaled bounds; lazy Pillow."""
    pixels, meta = _load_png(png_bytes, max_bytes, max_output_pixels, scale)
    pixels.close()
    return meta


def decode_png(png_bytes, max_output_pixels=DEFAULT_OUTPUT_PIXELS, scale=4, max_bytes=DEFAULT_PNG_BYTES):
    pixels, meta = _load_png(png_bytes, max_bytes, max_output_pixels, scale)
    np = dependency("numpy")
    array = np.asarray(pixels, dtype=np.uint8).copy()
    pixels.close()
    return {**meta, "rgb": array[:, :, :3].copy(), "alpha": array[:, :, 3].copy() if meta["channels"] == 4 else None}


def encode_png(rgb, alpha=None, max_pixels=DEFAULT_OUTPUT_PIXELS, max_bytes=32 * 1024 * 1024):
    np, Image = dependency("numpy"), dependency("PIL.Image")
    array = np.asarray(rgb)
    require(array.dtype == np.uint8 and array.ndim == 3 and array.shape[2] == 3
            and array.shape[0] > 0 and array.shape[1] > 0 and array.shape[0] * array.shape[1] <= max_pixels,
            "PNG RGB output must be bounded uint8 HWC")
    if alpha is not None:
        alpha = np.asarray(alpha)
        require(alpha.dtype == np.uint8 and alpha.shape == array.shape[:2], "PNG alpha shape/type mismatch")
        array = np.concatenate((array, alpha[:, :, None]), axis=2)
    with Image.fromarray(np.ascontiguousarray(array)) as image:
        stream = io.BytesIO()
        image.save(stream, format="PNG", compress_level=6)
        encoded = stream.getvalue()
    require(len(encoded) <= max_bytes, "encoded PNG output exceeds byte limit")
    return encoded


def media_sha256(blob):
    require(isinstance(blob, bytes), "media identity requires bytes")
    return hashlib.sha256(blob).hexdigest()


def validate_text(payload, max_bytes=64 * 1024):
    """Validate bounded UTF-8 text without importing any optional dependency."""
    require(type(max_bytes) is int and 1 <= max_bytes <= 16 * 1024 * 1024, "invalid text byte limit")
    require(isinstance(payload, (bytes, str)), "text must be UTF-8 bytes or a string")
    try:
        encoded = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("text must be valid UTF-8") from exc
    require(0 < len(encoded) <= max_bytes and text.strip(), "text must be nonempty within its byte limit")
    return {"text": text, "utf8_bytes": len(encoded)}

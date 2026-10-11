"""Real optional compact speech and trained camera-upscale adapters.

Optional model libraries are imported only by make_adapter. No device capture,
model downloads, or bicubic-only inference fallback is performed. GraphEngine
owns operation serialization and closes adapters only after active calls drain.
"""
import base64
from pathlib import Path, PurePosixPath
import re
import sys

import stream_media as media


class StreamCancelled(RuntimeError):
    """An operation stopped before publishing a result."""


def _check_cancel(cancel):
    if cancel is not None:
        predicate = cancel if callable(cancel) else getattr(cancel, "is_set", None)
        if not callable(predicate):
            raise TypeError("cancel must be callable or an Event")
        if predicate():
            raise StreamCancelled("stream operation cancelled")


def _asset(manifest, root, relative):
    media.require(isinstance(relative, str) and relative and "\\" not in relative and "\0" not in relative,
                  "model asset path must be a relative POSIX path")
    name = PurePosixPath(relative)
    media.require(not name.is_absolute() and all(part not in ("", ".", "..") for part in name.parts)
                  and ":" not in relative and str(name) == relative, "unsafe model asset path")
    media.require(relative in manifest["files"], "model asset is not covered by the verified manifest")
    path = (root / relative).resolve()
    media.require(root in path.parents and path.is_file(), "model asset is missing or outside its model directory")
    return path


def _manifest(manifest, model_dir):
    media.require(isinstance(manifest, dict) and manifest.get("schema") == "colibri.runtime-model/v1", "unsupported runtime model manifest")
    expected = {"whisper_openvino": "speech", "super_resolution_openvino": "camera_upscale", "text_openvino": "text_graph"}
    media.require(manifest.get("adapter") in expected and manifest.get("family") == expected[manifest["adapter"]], "manifest family/adapter mismatch")
    media.require(isinstance(manifest.get("model_id"), str) and manifest["model_id"], "model_id is required")
    media.require(isinstance(manifest.get("files"), dict) and manifest["files"], "manifest must list model assets")
    media.require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in manifest["files"].values()), "manifest assets need SHA256 identities")
    options = manifest.get("options", {})
    media.require(isinstance(options, dict), "adapter options must be an object")
    root = Path(model_dir).resolve()
    media.require(root.is_dir(), "model directory is missing")
    for name in manifest["files"]:
        _asset(manifest, root, name)
    return root, options


def _matches(requested, actual):
    return requested == actual or requested in ("GPU", "NPU") and actual == requested + ".0"


def _devices(core, requested):
    media.require(isinstance(requested, str) and (requested == "AUTO" or re.fullmatch(r"CPU|(?:GPU|NPU)(?:\.[0-9]+)?", requested)),
                  "device must be AUTO or explicit CPU/GPU/NPU[.id]")
    available = list(core.available_devices)
    if requested != "AUTO":
        media.require(any(_matches(requested, name) for name in available), f"requested device {requested!r} unavailable; available: {available}")
        return [requested]
    # A reproducible preference, not a measured fastest-device policy. Compile
    # failures can move AUTO to the next available explicit device.
    candidates = [name for platform in ("GPU", "NPU", "CPU") for name in available
                  if name.split(".", 1)[0] == platform and re.fullmatch(r"CPU|(?:GPU|NPU)(?:\.[0-9]+)?", name)]
    media.require(candidates, "no compatible OpenVINO device is available")
    return candidates


def _execution_devices(compiled, requested):
    devices = compiled.get_property("EXECUTION_DEVICES")
    media.require(isinstance(devices, (tuple, list)) and devices and all(isinstance(d, str) for d in devices), "compiled model has no verifiable execution devices")
    unique = sorted(set(devices))
    media.require(len(unique) == 1 and _matches(requested, unique[0]), f"model execution substituted device {requested!r}: {unique}")
    return unique


def _integer(options, key, default, low, high):
    value = options.get(key, default)
    media.require(type(value) is int and low <= value <= high, f"invalid {key}")
    return value


def _speech_gpu_restriction(candidate, genai):
    # Qualified CPU teardown succeeds; the full adapter's Windows GPU process
    # hangs after close/interpreter exit with this reproduced SDK release.
    # Restrict this speech lane rather than disabling qualified text/SR GPU.
    version = str(getattr(genai, "__version__", "unknown"))
    if sys.platform == "win32" and candidate.split(".", 1)[0] == "GPU" and version.startswith("2026.4.1"):
        return "Windows speech GPU is unavailable with OpenVINO GenAI 2026.4.1 pending clean process-shutdown qualification; use CPU"
    return None


class _Adapter:
    def _open(self):
        if self._closed:
            raise RuntimeError("model adapter is closed")

    def close(self):
        self._closed = True


class WhisperOpenVinoAdapter(_Adapter):
    def __init__(self, manifest, root, options, requested, ov):
        model_path = options.get("model_path", ".")
        media.require(isinstance(model_path, str) and model_path, "invalid Whisper model_path")
        if model_path == ".":
            pipeline_root, prefix = root, ""
        else:
            relative = PurePosixPath(model_path)
            media.require(not relative.is_absolute() and str(relative) == model_path
                          and "\\" not in model_path and ":" not in model_path and "\0" not in model_path
                          and all(part not in (".", "..") for part in relative.parts), "unsafe Whisper model_path")
            pipeline_root, prefix = (root / model_path).resolve(), model_path + "/"
            media.require(root in pipeline_root.parents and pipeline_root.is_dir(), "Whisper model_path is outside its model directory")
        for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                     "openvino_encoder_model.xml", "openvino_encoder_model.bin",
                     "openvino_decoder_model.xml", "openvino_decoder_model.bin",
                     "openvino_tokenizer.xml", "openvino_tokenizer.bin",
                     "openvino_detokenizer.xml", "openvino_detokenizer.bin"):
            _asset(manifest, root, prefix + name)
        self._max_bytes = _integer(manifest, "max_input_bytes", media.DEFAULT_WAV_BYTES, 44, 16 * 1024 * 1024)
        self._max_seconds = options.get("max_audio_seconds", 30)
        media.require(type(self._max_seconds) in (int, float) and 0 < self._max_seconds <= 30, "invalid max_audio_seconds")
        max_tokens = _integer(options, "max_new_tokens", 448, 1, 448)
        self._max_tokens = max_tokens
        candidates, attempts = _devices(ov.Core(), requested), []
        genai = media.dependency("openvino_genai")
        for candidate in candidates:
            restriction = _speech_gpu_restriction(candidate, genai)
            if restriction is not None:
                attempts.append({"device": candidate, "reason": restriction})
                if requested != "AUTO":
                    raise RuntimeError(restriction)
                continue
            try:
                pipe = genai.WhisperPipeline(str(pipeline_root), candidate)
                selected = candidate
                break
            except Exception as exc:
                attempts.append({"device": candidate, "error": str(exc)})
                if requested != "AUTO":
                    raise
        else:
            raise RuntimeError(f"Whisper compilation failed on available devices: {attempts}")
        self._pipe, self._closed = pipe, False
        self.info = {"adapter": "whisper_openvino", "backend": "openvino_genai", "device": selected,
                     "requested_device": requested, "precision": manifest.get("precision", "unknown"),
                     "runtime_precision": "unknown", "execution_devices_verified": False,
                     "device_source": "explicit-genai-constructor", "auto_attempts": attempts,
                     "model_id": manifest["model_id"]}

    def transcribe(self, wav_bytes, *, task="transcribe", language="en", cancel=None):
        self._open()
        media.require(task in ("transcribe", "translate") and language in (None, "en"), "speech task/language must be transcribe|translate and en|None")
        _check_cancel(cancel)
        audio = media.decode_wav(wav_bytes, max_audio_seconds=self._max_seconds, max_bytes=self._max_bytes)
        _check_cancel(cancel)

        cancellation_seen = False
        cancel_holder = [cancel]

        def streamer(_text):
            nonlocal cancellation_seen
            try:
                _check_cancel(cancel_holder[0] if cancel_holder else None)
                return False
            except StreamCancelled:
                cancellation_seen = True
                return True  # Supported short-form GenAI token-boundary stop.

        generation = {"task": task, "return_timestamps": False,
                      "max_new_tokens": self._max_tokens, "streamer": streamer}
        if language == "en":
            generation["language"] = "<|en|>"
        else:
            # Current GenAI rejects None in arbitrary keyword properties. Its
            # typed generation config supports language=None for auto-detect.
            get_config = getattr(self._pipe, "get_generation_config", None)
            if callable(get_config):
                config = get_config()
                config.language = None
                generation["generation_config"] = config
        try:
            result = self._pipe.generate(audio["samples"].tolist(), **generation)
        finally:
            # GenAI may retain the Python streamer after generate returns. Do
            # not let that callback retain a GraphEngine/session cancellation
            # closure, which creates a native/Python lifetime cycle at shutdown.
            cancel_holder.clear()
        if cancellation_seen:
            raise StreamCancelled("stream operation cancelled")
        _check_cancel(cancel)
        texts = getattr(result, "texts", None)
        media.require(isinstance(texts, (list, tuple)) and len(texts) == 1 and isinstance(texts[0], str), "Whisper returned malformed decoded text")
        return {"text": texts[0], "segments": [], "duration": audio["duration"], "language": language,
                "target_language": "en" if task == "translate" or language == "en" else None,
                "task": task, "backend": self.info["backend"], "device": self.info["device"],
                "execution_devices_verified": False}

    def close(self):
        super().close()
        self._pipe = None


class SuperResolutionOpenVinoAdapter(_Adapter):
    """The trained model uses BGR 0..255 LR+bicubic inputs, normalized output."""
    SCALE = 4

    def __init__(self, manifest, root, options, requested, ov):
        xml_name = options.get("model_xml")
        if xml_name is None:
            names = [name for name in manifest["files"] if name.endswith("single-image-super-resolution-1032.xml")]
            media.require(len(names) == 1, "camera manifest must identify model_xml")
            xml_name = names[0]
        xml = _asset(manifest, root, xml_name)
        binary = _asset(manifest, root, str(PurePosixPath(xml_name).with_suffix(".bin")))
        self._np, self._Image = media.dependency("numpy"), media.dependency("PIL.Image")
        self._max_input = _integer(manifest, "max_input_bytes", media.DEFAULT_PNG_BYTES, 45, 32 * 1024 * 1024)
        self._max_pixels = _integer(options, "max_output_pixels", media.DEFAULT_OUTPUT_PIXELS, 1, media.MAX_OUTPUT_PIXELS)
        self._max_output = _integer(options, "max_output_bytes", 32 * 1024 * 1024, 1024, 128 * 1024 * 1024)
        precision_hint = options.get("inference_precision", "f32")
        media.require(precision_hint in ("f32", "f16"), "inference_precision must be f32 or f16 (floating-point hints)")
        core = ov.Core()
        candidates = _devices(core, requested)
        model = core.read_model(model=str(xml), weights=str(binary))
        media.require(len(model.inputs) == 2 and len(model.outputs) == 1, "camera model must have two inputs and one output")
        self._low_port, self._high_port = model.input("0"), model.input("1")
        low, high, output = list(self._low_port.shape), list(self._high_port.shape), list(model.output(0).shape)
        media.require(len(low) == 4 and low[:2] == [1, 3] and 1 <= low[2] <= 1024 and 1 <= low[3] <= 1024
                      and low[2] * low[3] <= 270 * 480
                      and high == output == [1, 3, low[2] * self.SCALE, low[3] * self.SCALE], "camera model requires static NCHW 4x super-resolution shapes")
        self._tile_h, self._tile_w = low[2], low[3]
        default_overlap = min(16, (min(self._tile_h, self._tile_w) - 1) // 2)
        self._overlap = _integer(options, "tile_overlap", default_overlap, 0, (min(self._tile_h, self._tile_w) - 1) // 2)
        attempts = []
        for candidate in candidates:
            try:
                platform = candidate.split(".", 1)[0]
                media.require(platform in ("CPU", "GPU") or precision_hint == "f32", "f16 inference hints require a CPU/GPU plugin")
                config = {"INFERENCE_PRECISION_HINT": getattr(ov.Type, precision_hint)} if platform in ("CPU", "GPU") else {}
                compiled = core.compile_model(model=model, device_name=candidate, config=config)
                executed = _execution_devices(compiled, candidate)
                selected = candidate
                break
            except Exception as exc:
                attempts.append({"device": candidate, "error": str(exc)})
                if requested != "AUTO":
                    raise
        else:
            raise RuntimeError(f"camera model compilation failed on available devices: {attempts}")
        self._compiled, self._request = compiled, compiled.create_infer_request()
        self._output_port = compiled.output(0)
        self._closed = False
        self.info = {"adapter": "super_resolution_openvino", "backend": "openvino", "device": executed[0],
                     "compiled_device": selected, "requested_device": requested, "execution_devices_verified": True,
                     "precision": manifest.get("precision", "unknown"), "runtime_precision": "unknown",
                     "inference_precision_hint": precision_hint if selected.split(".", 1)[0] in ("CPU", "GPU") else "plugin-default",
                     "scale": self.SCALE, "auto_attempts": attempts, "model_id": manifest["model_id"]}

    def upscale(self, png_bytes, *, cancel=None):
        self._open()
        _check_cancel(cancel)
        image = media.decode_png(png_bytes, max_output_pixels=self._max_pixels, scale=self.SCALE, max_bytes=self._max_input)
        np, Image = self._np, self._Image
        height, width = image["height"], image["width"]
        output = np.empty((height * self.SCALE, width * self.SCALE, 3), dtype=np.uint8)
        halo, step_h, step_w = self._overlap, self._tile_h - 2 * self._overlap, self._tile_w - 2 * self._overlap
        # Edge padding provides context while every original pixel retains its
        # own spatial coordinate. Overlap is cropped, not duplicated or resized.
        for top in range(0, height, step_h):
            for left in range(0, width, step_w):
                _check_cancel(cancel)
                start_y, start_x = top - halo, left - halo
                end_y, end_x = start_y + self._tile_h, start_x + self._tile_w
                tile = image["rgb"][max(0, start_y):min(height, end_y), max(0, start_x):min(width, end_x)]
                # Pad one fixed-size tile, not the entire image: extremely
                # narrow/wide inputs otherwise cause aspect-ratio memory growth.
                tile = np.pad(tile, ((max(0, -start_y), max(0, end_y - height)),
                                     (max(0, -start_x), max(0, end_x - width)), (0, 0)), mode="edge")
                bgr = tile[:, :, ::-1]
                with Image.fromarray(np.ascontiguousarray(tile)) as tile_image:
                    bicubic = np.asarray(tile_image.resize((self._tile_w * self.SCALE, self._tile_h * self.SCALE), Image.Resampling.BICUBIC))[:, :, ::-1]
                inputs = {"0": np.ascontiguousarray(bgr.transpose(2, 0, 1)[None], dtype=np.float32),
                          "1": np.ascontiguousarray(bicubic.transpose(2, 0, 1)[None], dtype=np.float32)}
                scores = np.asarray(self._request.infer(inputs)[self._output_port])
                _check_cancel(cancel)
                media.require(scores.shape == (1, 3, self._tile_h * self.SCALE, self._tile_w * self.SCALE)
                              and scores.dtype.kind == "f" and bool(np.all(np.isfinite(scores))), "camera model returned invalid/nonfinite output")
                rgb = (np.clip(scores[0].transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
                take_h, take_w = min(step_h, height - top), min(step_w, width - left)
                crop = rgb[halo * self.SCALE:(halo + take_h) * self.SCALE,
                           halo * self.SCALE:(halo + take_w) * self.SCALE]
                output[top * self.SCALE:(top + take_h) * self.SCALE, left * self.SCALE:(left + take_w) * self.SCALE] = crop
        alpha = None
        if image["alpha"] is not None:
            with Image.fromarray(image["alpha"]) as alpha_image:
                alpha = np.asarray(alpha_image.resize((width * self.SCALE, height * self.SCALE), Image.Resampling.BICUBIC)).copy()
        _check_cancel(cancel)
        executed = _execution_devices(self._compiled, self.info["compiled_device"])
        encoded = media.encode_png(output, alpha, max_pixels=self._max_pixels, max_bytes=self._max_output)
        _check_cancel(cancel)
        return {"png_base64": base64.b64encode(encoded).decode("ascii"), "width": width * self.SCALE,
                "height": height * self.SCALE, "scale": self.SCALE, "backend": self.info["backend"],
                "device": executed[0], "execution_devices_verified": True}

    def close(self):
        super().close()
        self._request = self._compiled = None


class TextOpenVinoAdapter(_Adapter):
    """Stateless complete/chat calls on a real compact GenAI text model."""
    def __init__(self, manifest, root, options, requested, ov):
        for name in ("config.json", "openvino_model.xml", "openvino_model.bin",
                     "openvino_tokenizer.xml", "openvino_tokenizer.bin",
                     "openvino_detokenizer.xml", "openvino_detokenizer.bin"):
            _asset(manifest, root, name)
        self._max_input = _integer(manifest, "max_input_bytes", 64 * 1024, 1, 16 * 1024 * 1024)
        self._max_new_tokens = _integer(options, "max_tokens", 256, 1, 256)
        self._max_prompt = _integer(options, "max_prompt_tokens", 2048, 1, 4096)
        self._max_context = _integer(options, "max_context_tokens", 4096, 2, 4096)
        candidates, attempts = _devices(ov.Core(), requested), []
        genai = media.dependency("openvino_genai")
        for candidate in candidates:
            try:
                pipe = genai.LLMPipeline(str(root), candidate)
                tokenizer = pipe.get_tokenizer()
                media.require(callable(getattr(tokenizer, "encode", None)), "text backend cannot enforce prompt-token limits")
                selected = candidate
                break
            except Exception as exc:
                attempts.append({"device": candidate, "error": str(exc)})
                if requested != "AUTO":
                    raise
        else:
            raise RuntimeError(f"text compilation failed on available devices: {attempts}")
        self._pipe, self._tokenizer, self._closed = pipe, tokenizer, False
        self.info = {"adapter": "text_openvino", "backend": "openvino_genai", "device": selected,
                     "requested_device": requested, "precision": manifest.get("precision", "unknown"),
                     "runtime_precision": "unknown", "execution_devices_verified": False,
                     "device_source": "explicit-genai-constructor", "auto_attempts": attempts,
                     "model_id": manifest["model_id"], "stateful_chat": False,
                     "max_new_tokens": self._max_new_tokens, "max_prompt_tokens": self._max_prompt}

    def complete(self, prompt, *, max_new_tokens=None, cancel=None):
        self._open()
        _check_cancel(cancel)
        text = media.validate_text(prompt, self._max_input)["text"]
        tokens = min(128, self._max_new_tokens) if max_new_tokens is None else max_new_tokens
        media.require(type(tokens) is int and 1 <= tokens <= self._max_new_tokens, "max_new_tokens exceeds the compact text limit")
        encoded = self._tokenizer.encode(text)
        shape = getattr(getattr(encoded, "input_ids", None), "shape", None)
        media.require(shape is not None and len(shape) == 2 and shape[0] == 1 and 0 < shape[1] <= self._max_prompt,
                      "prompt-token count is unavailable or exceeds its limit")
        prompt_count = int(shape[1])
        media.require(prompt_count + tokens <= self._max_context, "prompt plus generation exceeds context limit")
        _check_cancel(cancel)
        cancellation_seen = False
        cancel_holder = [cancel]

        def streamer(_text):
            nonlocal cancellation_seen
            try:
                _check_cancel(cancel_holder[0] if cancel_holder else None)
                return False
            except StreamCancelled:
                cancellation_seen = True
                return True

        # No start_chat, shared message history, or retained client prompt is
        # used. Each call supplies the entire raw/rendered prompt independently.
        try:
            result = self._pipe.generate(text, max_new_tokens=tokens, do_sample=False,
                                         num_beams=1, num_return_sequences=1, streamer=streamer)
        finally:
            cancel_holder.clear()
        if cancellation_seen:
            raise StreamCancelled("stream operation cancelled")
        _check_cancel(cancel)
        if not isinstance(result, str):
            texts = getattr(result, "texts", None)
            media.require(isinstance(texts, (tuple, list)) and len(texts) == 1 and isinstance(texts[0], str), "text model returned malformed output")
            result = texts[0]
        return {"text": result, "backend": self.info["backend"], "device": self.info["device"],
                "execution_devices_verified": False, "prompt_token_count": prompt_count,
                "max_new_tokens": tokens, "stateful_chat": False}

    def chat(self, messages, *, max_new_tokens=None, cancel=None):
        self._open()
        _check_cancel(cancel)
        media.require(isinstance(messages, list) and 1 <= len(messages) <= 64, "chat messages must be a bounded nonempty list")
        rendered_inputs, total_bytes = [], 0
        for message in messages:
            media.require(isinstance(message, dict) and message.get("role") in ("system", "user", "assistant")
                          and isinstance(message.get("content"), str), "chat messages need valid role/content strings")
            value = media.validate_text(message["content"], self._max_input)
            total_bytes += value["utf8_bytes"]
            media.require(total_bytes <= self._max_input, "chat content exceeds byte limit")
            rendered_inputs.append({"role": message["role"], "content": value["text"]})
        template = getattr(self._tokenizer, "apply_chat_template", None)
        media.require(callable(template), "model tokenizer does not provide a chat template")
        rendered = template(rendered_inputs, add_generation_prompt=True)
        media.require(isinstance(rendered, str), "chat template did not return a string")
        return self.complete(rendered, max_new_tokens=max_new_tokens, cancel=cancel)

    def close(self):
        super().close()
        self._pipe = self._tokenizer = None


def make_adapter(manifest, model_dir, device="AUTO"):
    """Create one actual backend after GraphEngine verifies manifest file hashes."""
    root, options = _manifest(manifest, model_dir)
    ov = media.dependency("openvino")
    if manifest["adapter"] == "whisper_openvino":
        return WhisperOpenVinoAdapter(manifest, root, options, device, ov)
    if manifest["adapter"] == "text_openvino":
        return TextOpenVinoAdapter(manifest, root, options, device, ov)
    return SuperResolutionOpenVinoAdapter(manifest, root, options, device, ov)

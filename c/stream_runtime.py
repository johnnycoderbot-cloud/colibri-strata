"""Persistent graph adapters and bounded audio/camera stream sessions.

Memory admission uses manifest estimates and queue byte bounds, not physical
reservations. Backends run synchronously under one engine lock. Accepted audio
is retained until polled or explicitly cancelled; camera replacement drops
only pending frames and reports their identities. No accelerator is simulated.
"""
from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
import unicodedata
import uuid


MANIFEST_NAME = "colibri_runtime.json"
MANIFEST_SCHEMA = "colibri.runtime-model/v1"
MAX_MANIFEST_BYTES = 65536
MAX_INPUT_BYTES = 64 << 20
MAX_OUTPUT_BYTES = 128 << 20
MAX_MEMORY_BYTES = 64 << 30
MAX_QUEUE_CAPACITY = 16
_PAIRS = {"speech": ("whisper_openvino", "audio"),
          "camera_upscale": ("super_resolution_openvino", "camera"),
          "text_graph": ("text_openvino", "text")}


class ManifestError(ValueError):
    """A model manifest or its declared assets failed validation."""


class GraphCancelled(RuntimeError):
    """Cancellation completed without publishing an obsolete result."""


def _integer(value, name, minimum=0, maximum=MAX_MEMORY_BYTES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value, name, minimum=0, maximum=2**53 - 1):
    if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return value


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ManifestError(f"nonfinite JSON constant: {value}")


def _json_copy(value, limit, name):
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=True,
                             separators=(",", ":")).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    if len(encoded) > limit:
        raise ValueError(f"{name} exceeds its {limit}-byte bound")
    return json.loads(encoded)


def _asset_parts(name):
    if not isinstance(name, str) or not name or len(name) > 1024 or name.startswith("/"):
        raise ManifestError("asset paths must be nonempty relative paths")
    if any(ord(c) < 32 or c in '\\<>:"|?*' for c in name):
        raise ManifestError(f"unsafe asset path: {name!r}")
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
        f"LPT{i}" for i in range(1, 10)}
    for part in parts:
        if not part or part in (".", "..") or part.endswith((".", " ")):
            raise ManifestError(f"unsafe asset path: {name!r}")
        if part.split(".")[0].upper() in reserved:
            raise ManifestError(f"reserved asset path: {name!r}")
    return parts


def _checked_file(root, parts, max_bytes):
    current = root
    for index, part in enumerate(parts):
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ManifestError(f"symlink asset refused: {current}")
        if index + 1 < len(parts) and not stat.S_ISDIR(metadata.st_mode):
            raise ManifestError(f"asset parent is not a directory: {current}")
    if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= max_bytes:
        raise ManifestError(f"asset must be a nonempty bounded regular file: {current}")
    current.resolve(strict=True).relative_to(root)
    return current, metadata


def _read_checked(path, before, limit, *, digest=False):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if not stat.S_ISREG(opened.st_mode) or identity != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ManifestError(f"asset changed while opening: {path}")
        hasher, chunks, total = hashlib.sha256() if digest else None, [], 0
        while True:
            chunk = os.read(fd, min(1 << 20, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ManifestError(f"asset exceeds bound: {path}")
            if digest:
                hasher.update(chunk)
            else:
                chunks.append(chunk)
        after = os.fstat(fd)
        if total != before.st_size or identity != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ManifestError(f"asset changed during validation: {path}")
        return hasher.hexdigest() if digest else b"".join(chunks)
    finally:
        os.close(fd)


def read_manifest_metadata(model_dir):
    """Read bounded descriptor metadata; assets are explicitly NOT verified.

    Registry lookups must not impose an inference memory budget or scan model
    weights. GraphEngine uses validate_manifest for full asset/hash admission.
    """
    try:
        root = Path(model_dir)
        if root.is_symlink() or not root.is_dir():
            raise ManifestError("model directory must be a real directory")
        root = root.resolve(strict=True)
        path, metadata = _checked_file(root, [MANIFEST_NAME], MAX_MANIFEST_BYTES)
        raw = _read_checked(path, metadata, MAX_MANIFEST_BYTES)
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                              parse_constant=_reject_constant)
        # Encoding without ASCII escapes rejects lone surrogate code points.
        json.dumps(manifest, allow_nan=False, ensure_ascii=False).encode("utf-8", "strict")
        manifest = _json_copy(manifest, MAX_MANIFEST_BYTES, "manifest")
        if not isinstance(manifest, dict) or not {"schema", "family", "model_id"} <= manifest.keys():
            raise ManifestError("runtime metadata is missing descriptor fields")
        if manifest["schema"] != MANIFEST_SCHEMA:
            raise ManifestError("unsupported runtime manifest schema")
        if not isinstance(manifest["family"], str) or manifest["family"] not in _PAIRS:
            raise ManifestError("unsupported runtime metadata family")
        if not isinstance(manifest["model_id"], str) or not re.fullmatch(r"[!-~]{1,128}", manifest["model_id"]):
            raise ManifestError("model_id must be nonempty printable ASCII without spaces")
        return manifest
    except ManifestError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise ManifestError(f"invalid runtime metadata: {error}") from error


def validate_manifest(model_dir, memory_budget_bytes=1 << 30):
    """Validate every asset before any optional model backend is imported.

    Extra bounded JSON metadata is retained for provenance/shape information.
    Duplicate keys, path/inode aliases, symlinks and nonregular files fail closed.
    This validation assumes the caller keeps its model directory immutable
    after admission; adapters also recheck their declared model asset paths.
    """
    _integer(memory_budget_bytes, "memory_budget_bytes", 1)
    try:
        root = Path(model_dir)
        if root.is_symlink() or not root.is_dir():
            raise ManifestError("model directory must be a real directory")
        root = root.resolve(strict=True)
        manifest = read_manifest_metadata(root)
        required = {"schema", "family", "model_id", "adapter", "files", "precision",
                    "max_input_bytes", "resident_bytes", "workspace_bytes", "session_bytes", "options"}
        if not isinstance(manifest, dict) or not required <= manifest.keys():
            raise ManifestError("runtime manifest is missing required fields")
        family = manifest["family"]
        if not isinstance(family, str) or family not in _PAIRS or manifest["adapter"] != _PAIRS[family][0]:
            raise ManifestError("unsupported family/adapter pair")
        if not isinstance(manifest["precision"], str) or not re.fullmatch(r"[ -~]{1,64}", manifest["precision"]):
            raise ManifestError("precision must be a bounded ASCII description")
        _integer(manifest["max_input_bytes"], "max_input_bytes", 1, MAX_INPUT_BYTES)
        for name in ("resident_bytes", "workspace_bytes", "session_bytes"):
            _integer(manifest[name], name)
        if not isinstance(manifest["options"], dict):
            raise ManifestError("options must be an object")
        files = manifest["files"]
        if not isinstance(files, dict) or not 1 <= len(files) <= 1024:
            raise ManifestError("files must be a nonempty bounded object")
        seen_names, seen_inodes, assets, total = set(), set(), [], 0
        for name, digest in files.items():
            parts = _asset_parts(name)
            normalized = unicodedata.normalize("NFC", name).casefold()
            if normalized in seen_names or normalized == MANIFEST_NAME.casefold():
                raise ManifestError("duplicate/recursive asset identity")
            seen_names.add(normalized)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ManifestError(f"invalid SHA256 for asset: {name}")
            path, meta = _checked_file(root, parts, memory_budget_bytes)
            inode = (meta.st_dev, meta.st_ino)
            if meta.st_ino and inode in seen_inodes:
                raise ManifestError("hard-linked asset aliases are not permitted")
            seen_inodes.add(inode)
            total += meta.st_size
            if total > memory_budget_bytes:
                raise MemoryError("model assets exceed the memory estimate budget")
            assets.append((path, meta, digest))
        base = max(total, manifest["resident_bytes"]) + manifest["workspace_bytes"]
        if base > memory_budget_bytes:
            raise MemoryError("resident/workspace estimates exceed memory budget")
        for path, meta, digest in assets:
            if _read_checked(path, meta, memory_budget_bytes, digest=True) != digest:
                raise ManifestError(f"SHA256 mismatch for asset: {path.name}")
            if path.resolve(strict=True).is_relative_to(root) is False:
                raise ManifestError("asset escaped model directory during validation")
        return manifest, root, total, base
    except (ManifestError, MemoryError):
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise ManifestError(f"invalid runtime model: {error}") from error


@dataclass(frozen=True, slots=True)
class _Frame:
    sequence: int
    timestamp_ms: float
    payload: bytes
    final: bool
    request_id: str

    def identity(self):
        return {"sequence": self.sequence, "timestamp_ms": self.timestamp_ms,
                "request_id": self.request_id, "final": self.final}


class _Stream:
    def __init__(self, engine, kind, capacity, options, reservation):
        self.engine, self.kind, self.capacity, self.options = engine, kind, capacity, options
        self.id, self.reservation = uuid.uuid4().hex, reservation
        self.condition = threading.Condition()
        self.pending, self.events = deque(), deque()
        self.event_capacity = capacity + 2
        self.cancelled = threading.Event()
        self.generation, self.last_sequence, self.last_timestamp = 0, -1, -1
        self.final_accepted, self.state, self.in_flight = False, "open", None
        self.dropped, self.accepted = 0, 0
        self.worker = threading.Thread(target=self._work, name=f"colibri-{kind}-{self.id[:8]}", daemon=True)

    def _publish(self, event, generation):
        with self.condition:
            while len(self.events) >= self.event_capacity and not self.cancelled.is_set():
                self.condition.wait()
            if self.cancelled.is_set() or self.generation != generation:
                return False
            event.update(stream_id=self.id, kind=self.kind, generation=generation)
            self.events.append(event)
            self.condition.notify_all()
            return True

    def _work(self):
        try:
            while True:
                with self.condition:
                    while not self.pending and not self.cancelled.is_set():
                        self.condition.wait()
                    if self.cancelled.is_set():
                        return
                    frame, generation = self.pending.popleft(), self.generation
                    self.in_flight = frame
                    self.condition.notify_all()
                try:
                    if self.kind == "audio":
                        result = self.engine.transcribe(frame.payload, cancel=self.cancelled.is_set, **self.options)
                        for segment in result.get("segments", []):
                            for key in ("start", "end"):
                                if key in segment:
                                    segment[key] = _number(segment[key], f"segment {key}") + frame.timestamp_ms / 1000
                        result = _json_copy(result, self.engine._output_bound, "timestamped result")
                    else:
                        result = self.engine.upscale(frame.payload, cancel=self.cancelled.is_set)
                    event = {"type": "result", **frame.identity(), "result": result}
                except Exception as error:
                    event = {"type": "error", **frame.identity(), "error": str(error)[:2048],
                             "error_type": type(error).__name__}
                with self.condition:
                    self.in_flight = None
                    self.condition.notify_all()
                if not self._publish(event, generation):
                    return
                if frame.final:
                    if self._publish({"type": "finished", **frame.identity()}, generation):
                        with self.condition:
                            if self.generation == generation and not self.cancelled.is_set():
                                self.state = "finished"
                            self.condition.notify_all()
                    return
        finally:
            with self.condition:
                self.in_flight = None
                self.condition.notify_all()

    def cancel(self):
        with self.condition:
            if self.cancelled.is_set():
                return {"stream_id": self.id, "state": "cancelled", "generation": self.generation}
            discarded = [frame.identity() for frame in self.pending]
            active = self.in_flight.identity() if self.in_flight else None
            self.generation += 1
            self.cancelled.set()
            self.pending.clear()
            self.events.clear()
            self.state = "cancelled"
            receipt = {"stream_id": self.id, "kind": self.kind, "state": "cancelled",
                       "generation": self.generation, "discarded": discarded, "in_flight": active}
            self.events.append({"type": "cancelled", **receipt})
            self.condition.notify_all()
            return receipt


class GraphEngine:
    runtime_kind = "graph"

    def __init__(self, model_dir, device="AUTO", memory_budget_bytes=1 << 30,
                 max_sessions=8, adapter_factory=None):
        _integer(max_sessions, "max_sessions", 1, 128)
        if not isinstance(device, str) or not re.fullmatch(r"[!-~]{1,128}", device):
            raise ValueError("device must be nonempty bounded ASCII")
        manifest, root, asset_bytes, base_bytes = validate_manifest(model_dir, memory_budget_bytes)
        self._manifest, self.model_dir, self.model_id = manifest, root, manifest["model_id"]
        self.modality, self.device = _PAIRS[manifest["family"]][1], device
        self._budget, self._base, self._asset_bytes = memory_budget_bytes, base_bytes, asset_bytes
        self._max_sessions, self._reserved = max_sessions, 0
        default_output = 1 << 20 if self.modality in ("audio", "text") else min(
            MAX_OUTPUT_BYTES, max(1 << 20, manifest["max_input_bytes"] * 16))
        self._output_bound = _integer(manifest["options"].get("max_output_bytes", default_output),
                                      "max_output_bytes", 1, MAX_OUTPUT_BYTES)
        self._state_lock, self._operation_lock = threading.RLock(), threading.Lock()
        self._close_lock, self._operation_owner, self._close_owner = threading.Lock(), None, None
        self._shutdown, self._sessions = threading.Event(), {}
        self._closing, self._closed = False, False
        self._max_text_tokens = 256
        if self.modality == "text":
            self._max_text_tokens = _integer(manifest["options"].get("max_tokens", 256), "max_tokens", 1, 256)
            _integer(manifest["options"].get("max_prompt_tokens", 2048), "max_prompt_tokens", 1, 4096)
            _integer(manifest["options"].get("max_context_tokens", 4096), "max_context_tokens", 2, 4096)
        if adapter_factory is None:
            from stream_models import make_adapter
            adapter_factory = make_adapter
        if not callable(adapter_factory):
            raise ValueError("adapter_factory must be callable")
        adapter = adapter_factory(copy.deepcopy(manifest), root, device)
        try:
            info = getattr(adapter, "info", None)
            if not isinstance(info, dict) or not all(isinstance(info.get(k), str) and info[k]
                                                     for k in ("backend", "device")):
                raise ValueError("adapter must report actual backend/device information")
            operation = {"audio": "transcribe", "camera": "upscale", "text": "complete"}[self.modality]
            if not callable(getattr(adapter, operation, None)):
                raise ValueError(f"adapter has no {operation} operation")
            self._backend_info = _json_copy(info, MAX_MANIFEST_BYTES, "backend info")
            self._device_evidence(self._backend_info)
            if self.modality == "text" and not callable(getattr(adapter, "chat", None)):
                raise ValueError("text adapter has no chat operation")
        except Exception:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
            raise
        self._adapter = adapter

    def _device_evidence(self, info, compiled_default=None):
        if not isinstance(info.get("device"), str) or not info["device"]:
            raise ValueError("backend result must identify its selected device")
        verified = info.get("execution_devices_verified", False)
        if type(verified) is not bool:
            raise ValueError("execution_devices_verified must be a boolean")
        info["requested_device"] = self.device
        info.setdefault("compiled_device", compiled_default or info["device"])
        if not isinstance(info["compiled_device"], str) or not info["compiled_device"]:
            raise ValueError("compiled_device must be a nonempty string")
        info["execution_devices_verified"] = verified
        # A constructor's selected device alone is not execution evidence.
        info["executed_device"] = info["device"] if verified else None

    def info(self):
        with self._state_lock:
            return {**copy.deepcopy(self._backend_info), "runtime_kind": self.runtime_kind,
                    "model_id": self.model_id, "family": self._manifest["family"],
                    "modality": self.modality, "precision": self._manifest["precision"],
                    "max_input_bytes": self._manifest["max_input_bytes"],
                    "max_output_bytes": self._output_bound,
                    "memory_budget_bytes": self._budget, "asset_bytes": self._asset_bytes,
                    "base_estimated_bytes": self._base, "session_estimated_bytes": self._reserved,
                    "memory_estimates_only": True, "active_sessions": len(self._sessions),
                    "max_sessions": self._max_sessions, "closing": self._closing, "closed": self._closed}

    def _input(self, payload, kind):
        if kind != self.modality:
            raise ValueError(f"this model supports {self.modality}, not {kind}")
        if type(payload) is not bytes or not 0 < len(payload) <= self._manifest["max_input_bytes"]:
            raise ValueError("payload must be nonempty immutable bytes within model bounds")
        from stream_media import validate_wav, validate_png
        options = self._manifest["options"]
        if kind == "audio":
            validate_wav(payload, max_audio_seconds=options.get("max_audio_seconds", 30),
                         max_bytes=self._manifest["max_input_bytes"])
        else:
            validate_png(payload, max_output_pixels=options.get("max_output_pixels", 5592405),
                         scale=options.get("scale", 4), max_bytes=self._manifest["max_input_bytes"])

    @staticmethod
    def _audio_options(task, language):
        if task not in ("transcribe", "translate") or (language is not None and language != "en"):
            raise ValueError("task must be transcribe/translate and language en/None")
        return {"task": task, "language": language}

    def _run(self, operation, payload, cancel, **options):
        predicate = cancel if callable(cancel) else getattr(cancel, "is_set", None)
        if cancel is not None and not callable(predicate):
            raise ValueError("cancel must be a callable or an Event")
        def cancelled():
            return self._shutdown.is_set() or (cancel is not None and bool(predicate()))
        if cancelled():
            raise GraphCancelled("request cancelled")
        with self._operation_lock:
            try:
                self._operation_owner = threading.get_ident()
                if cancelled():
                    raise GraphCancelled("request cancelled")
                try:
                    result = getattr(self._adapter, operation)(payload, cancel=cancelled, **options)
                except Exception as error:
                    if cancelled():
                        raise GraphCancelled("request cancelled") from error
                    raise
                if cancelled():
                    raise GraphCancelled("request cancelled")
                if not isinstance(result, dict):
                    raise ValueError("adapter result must be an object")
                result = dict(result)
                result.setdefault("backend", self._backend_info["backend"])
                result.setdefault("device", self._backend_info["device"])
                result.setdefault("model_id", self.model_id)
                result.setdefault("execution_devices_verified", self._backend_info["execution_devices_verified"])
                self._device_evidence(result, self._backend_info["compiled_device"])
                result = _json_copy(result, self._output_bound, "adapter result")
                if operation == "transcribe":
                    if not isinstance(result.get("text"), str) or not isinstance(result.get("segments", []), list):
                        raise ValueError("speech result requires text and a segments list")
                    if not all(isinstance(s, dict) for s in result.get("segments", [])):
                        raise ValueError("speech segments must be objects")
                    result.setdefault("segments", [])
                elif operation == "upscale" and not isinstance(result.get("png_base64"), str):
                    raise ValueError("camera result requires png_base64")
                elif operation in ("complete", "chat") and not isinstance(result.get("text"), str):
                    raise ValueError("text result requires text")
                if cancelled():
                    raise GraphCancelled("request cancelled")
                return result
            finally:
                self._operation_owner = None

    def transcribe(self, wavbytes, task="transcribe", language="en", cancel=None):
        options = self._audio_options(task, language)
        self._input(wavbytes, "audio")
        return self._run("transcribe", wavbytes, cancel, **options)

    def upscale(self, pngbytes, cancel=None):
        self._input(pngbytes, "camera")
        return self._run("upscale", pngbytes, cancel)

    def _text_tokens(self, max_new_tokens):
        if self.modality != "text":
            raise ValueError("this model does not support text generation")
        if max_new_tokens is not None:
            _integer(max_new_tokens, "max_new_tokens", 1, self._max_text_tokens)
        return {"max_new_tokens": max_new_tokens}

    def complete(self, prompt, max_new_tokens=None, cancel=None):
        options = self._text_tokens(max_new_tokens)
        from stream_media import validate_text
        text = validate_text(prompt, max_bytes=self._manifest["max_input_bytes"])["text"]
        return self._run("complete", text, cancel, **options)

    def chat(self, messages, max_new_tokens=None, cancel=None):
        options = self._text_tokens(max_new_tokens)
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise ValueError("messages must be a nonempty list of at most 64 role/content objects")
        from stream_media import validate_text
        copied, total = [], 0
        for message in messages:
            if not isinstance(message, dict) or message.keys() != {"role", "content"} or message.get("role") not in (
                    "system", "user", "assistant") or not isinstance(message.get("content"), str):
                raise ValueError("messages require system/user/assistant roles and string content")
            value = validate_text(message["content"], max_bytes=self._manifest["max_input_bytes"])
            total += value["utf8_bytes"]
            if total > self._manifest["max_input_bytes"]:
                raise ValueError("chat content exceeds model byte bounds")
            copied.append({"role": message["role"], "content": value["text"]})
        return self._run("chat", copied, cancel, **options)

    def create_stream(self, kind, queue_capacity=2, options=None):
        _integer(queue_capacity, "queue_capacity", 1, MAX_QUEUE_CAPACITY)
        if kind not in ("audio", "camera") or kind != self.modality:
            raise ValueError("stream kind must match this model's modality")
        options = {} if options is None else _json_copy(options, 4096, "stream options")
        if not isinstance(options, dict):
            raise ValueError("stream options must be an object")
        if kind == "audio":
            if options.keys() - {"task", "language"}:
                raise ValueError("unsupported audio stream option")
            options = self._audio_options(options.get("task", "transcribe"), options.get("language", "en"))
        elif options:
            raise ValueError("camera scale/shape options belong to the model manifest")
        # Two extra result-sized slots cover JSON staging and the active result.
        reservation = self._manifest["session_bytes"] + (queue_capacity + 1) * self._manifest["max_input_bytes"] + (
            queue_capacity + 4) * self._output_bound + 16384
        with self._state_lock:
            if self._closing:
                raise RuntimeError("engine is closing")
            if len(self._sessions) >= self._max_sessions:
                raise BufferError("session limit reached")
            if self._base + self._reserved + reservation > self._budget:
                raise MemoryError("stream queue/session estimates exceed memory budget")
            stream = _Stream(self, kind, queue_capacity, options, reservation)
            self._sessions[stream.id] = stream
            self._reserved += reservation
            try:
                stream.worker.start()
            except Exception:
                del self._sessions[stream.id]
                self._reserved -= reservation
                raise
            return stream.id

    def _stream(self, stream_id):
        if not isinstance(stream_id, str):
            raise ValueError("stream_id must be a string")
        with self._state_lock:
            return self._sessions[stream_id]

    def push_stream(self, stream_id, sequence, timestamp_ms, payload, final=False):
        _integer(sequence, "sequence", 0, 2**63 - 1)
        _number(timestamp_ms, "timestamp_ms")
        if type(final) is not bool:
            raise ValueError("final must be a boolean")
        stream = self._stream(stream_id)
        self._input(payload, stream.kind)
        with stream.condition:
            if self._shutdown.is_set() or stream.state != "open" or stream.final_accepted:
                raise RuntimeError("stream no longer accepts frames")
            if sequence <= stream.last_sequence or timestamp_ms < stream.last_timestamp:
                raise ValueError("sequence must increase and timestamps must not decrease")
            full = len(stream.pending) >= stream.capacity
            if full and stream.kind == "audio":
                raise BufferError("audio queue full; retry the same frame after draining")
            if full and len(stream.events) >= stream.event_capacity:
                raise BufferError("camera event queue full; drain events before retrying")
            dropped = None
            if full:
                dropped = stream.pending.popleft()
                stream.dropped += 1
                stream.events.append({"type": "dropped", "reason": "superseded_pending_frame",
                                      "stream_id": stream.id, "kind": stream.kind,
                                      "generation": stream.generation, **dropped.identity(),
                                      "dropped_total": stream.dropped})
            frame = _Frame(sequence, timestamp_ms, payload, final, uuid.uuid4().hex)
            stream.pending.append(frame)
            stream.last_sequence, stream.last_timestamp = sequence, timestamp_ms
            stream.final_accepted = final
            stream.accepted += 1
            stream.condition.notify_all()
            return {"accepted": True, "stream_id": stream.id, "kind": stream.kind,
                    "generation": stream.generation, **frame.identity(), "queued": len(stream.pending),
                    "dropped": dropped.identity() if dropped else None, "dropped_total": stream.dropped}

    def stream_events(self, stream_id, max_events=32, wait_ms=0):
        _integer(max_events, "max_events", 1, 128)
        _integer(wait_ms, "wait_ms", 0, 60000)
        stream = self._stream(stream_id)
        deadline = time.monotonic() + wait_ms / 1000
        with stream.condition:
            while not stream.events and stream.state not in ("finished", "cancelled"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                stream.condition.wait(remaining)
            events = [stream.events.popleft() for _ in range(min(max_events, len(stream.events)))]
            stream.condition.notify_all()
            return events

    def cancel_stream(self, stream_id):
        return self._stream(stream_id).cancel()

    def close_stream(self, stream_id, timeout=None):
        if timeout is not None:
            _number(timeout, "timeout", maximum=3600)
        stream = self._stream(stream_id)
        stream.cancel()
        if threading.current_thread() is stream.worker:
            raise RuntimeError("stream cannot close itself from its backend callback")
        stream.worker.join(timeout)
        if stream.worker.is_alive():
            return False
        with self._state_lock:
            if self._sessions.pop(stream.id, None) is stream:
                self._reserved -= stream.reservation
        return True

    def close(self, timeout=None):
        """Cancel/drain before backend destruction; timeout retains ownership.

        A timeout bounds waiting for active operations, not an adapter's own
        close callback. Retrying close after False completes the same shutdown.
        """
        if timeout is not None:
            _number(timeout, "timeout", maximum=3600)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_lock:
            if self._closed:
                return True
            self._closing = True
            self._shutdown.set()
            streams = list(self._sessions.values())
        for stream in streams:
            stream.cancel()
        if self._operation_owner == threading.get_ident() or self._close_owner == threading.get_ident():
            return False
        remaining = -1 if deadline is None else max(0, deadline - time.monotonic())
        if not self._close_lock.acquire(timeout=remaining):
            return False
        self._close_owner = threading.get_ident()
        try:
            with self._state_lock:
                if self._closed:
                    return True
                streams = list(self._sessions.values())
            for stream in streams:
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                stream.worker.join(remaining)
                if stream.worker.is_alive():
                    return False
                with self._state_lock:
                    if self._sessions.pop(stream.id, None) is stream:
                        self._reserved -= stream.reservation
            remaining = -1 if deadline is None else max(0, deadline - time.monotonic())
            if not self._operation_lock.acquire(timeout=remaining):
                return False
            try:
                close = getattr(self._adapter, "close", None)
                if callable(close):
                    close()
                with self._state_lock:
                    self._closed = True
                return True
            finally:
                self._operation_lock.release()
        finally:
            self._close_owner = None
            self._close_lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

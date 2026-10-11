"""Standard-library transport for the native C graph executable.

No Python model SDK, tensor math, image resampling or neural fallback. Colibri
keeps its existing Python control plane; the subprocess owns inference.
"""
import base64
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
MAX_JSON = 65536
MAX_RESULT = 128 << 20
MAX_STDERR = 1 << 20


def _finite_json(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_JSON:
        raise ValueError("native JSON exceeds bounds")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate native result key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite native result")
    result = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("native result must be an object")
    def check(value, depth=0):
        if depth > 32 or isinstance(value, float) and not math.isfinite(value):
            raise ValueError("invalid native result depth or number")
        if isinstance(value, dict):
            for key, child in value.items():
                key.encode("utf-8", "strict")
                check(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                check(child, depth + 1)
        elif isinstance(value, str):
            value.encode("utf-8", "strict")
    check(result)
    return result


def _exact(stream, count):
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise ValueError("truncated native graph frame")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def native_binary():
    path = Path(os.environ.get("COLI_NATIVE_GRAPH", ROOT / ("native_graph.exe" if os.name == "nt" else "native_graph")))
    if not path.is_file():
        raise RuntimeError("native graph engine is not built; run python c/tools/build_native_graph.py")
    return path.resolve()


def sdk_environment(source=None):
    env = dict(os.environ if source is None else source)
    sdk = env.get("COLI_NATIVE_SDK")
    if sdk:
        root = Path(sdk).resolve()
        if os.name == "nt":
            release = root / "runtime/bin/intel64/Release"
            tbb = root / "runtime/3rdparty/tbb/bin"
            env.setdefault("COLI_OV_C_LIB", str(release / "openvino_c.dll"))
            env.setdefault("COLI_GENAI_C_LIB", str(release / "openvino_genai_c.dll"))
            env.setdefault("COLI_OV_TOKENIZERS_LIB", str(release / "openvino_tokenizers.dll"))
            env["PATH"] = os.pathsep.join((str(release), str(tbb), env.get("PATH", "")))
        else:
            libraries = root / "runtime/lib/intel64"
            env.setdefault("COLI_OV_C_LIB", str(libraries / "libopenvino_c.so"))
            env.setdefault("COLI_GENAI_C_LIB", str(libraries / "libopenvino_genai_c.so"))
            env.setdefault("COLI_OV_TOKENIZERS_LIB", str(libraries / "libopenvino_tokenizers.so"))
            env["LD_LIBRARY_PATH"] = os.pathsep.join((str(libraries), env.get("LD_LIBRARY_PATH", "")))
    return env


class _OwnedProcess:
    """Keep pipe operations and cleanup attached to one process generation."""
    def __init__(self, process):
        self.process = process
        self.stop = threading.Event()
        self.threads = []


def _device(value):
    if not isinstance(value, str) or not re.fullmatch(r"CPU|(?:GPU|NPU)(?:\.[0-9]+)?", value):
        raise ValueError("invalid native device identity")
    return value + ".0" if value in ("GPU", "NPU") else value


class NativeGraphAdapter:
    def __init__(self, manifest, root, device="AUTO", *, binary=None, environment=None,
                 startup_timeout=180, request_timeout=180, close_timeout=10):
        for value in (startup_timeout, request_timeout, close_timeout):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 600:
                raise ValueError("native process timeout must be finite in (0,600]")
        if not re.fullmatch(r"AUTO|CPU|(?:GPU|NPU)(?:\.[0-9]+)?", device):
            raise ValueError("device must be AUTO or explicit CPU/GPU/NPU[.id]")
        self._manifest = manifest
        self._root = Path(root).resolve()
        self._binary = Path(binary).resolve() if binary else native_binary()
        self._env = sdk_environment(environment)
        self._startup_timeout, self._request_timeout, self._close_timeout = startup_timeout, request_timeout, close_timeout
        self._lock = threading.RLock()
        self._process = None
        self._owner = None
        self._validator = None
        self._broken = False
        self._closed = False
        self._counter = 0
        self._threads = []
        self.stderr = []
        self._stderr_bytes = 0
        self._stderr_lock = threading.Lock()
        self._args = [str(self._binary), "--model", str(self._root), "--device", device,
                      "--memory-budget", str(manifest.get("_native_memory_budget", max(1 << 30, manifest["resident_bytes"] + manifest["workspace_bytes"])))]
        self._requested_device = device
        self._start()

    def _start(self, deadline=None, cancel=None):
        if self._closed:
            raise RuntimeError("native model adapter is closed")
        if self._owner is not None or self._validator is not None:
            raise RuntimeError("previous native process cleanup is still pending")
        startup_deadline = time.monotonic() + self._startup_timeout
        if deadline is not None:
            startup_deadline = min(startup_deadline, deadline)
        self._checkpoint(startup_deadline, cancel)
        self._frames = queue.Queue(maxsize=4)
        self._process = subprocess.Popen(self._args, env=self._env, cwd=ROOT, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._owner = _OwnedProcess(self._process)
        self._stop, self._threads = self._owner.stop, self._owner.threads
        self._threads.extend((threading.Thread(target=self._reader, args=(self._process,self._frames,self._stop), daemon=True),
                              threading.Thread(target=self._stderr_reader, args=(self._process,), daemon=True)))
        try:
            for thread in self._threads:
                thread.start()
            kind, result, body = self._next(startup_deadline, cancel)
            if kind != "READY" or body:
                raise ValueError("native graph startup did not publish READY")
            if (result.get("inference_runtime") != "native-c" or
                    result.get("backend") != "openvino-native-c" or
                    type(result.get("execution_devices_verified")) is not bool):
                raise ValueError("engine did not confirm native C inference")
            if _device(result.get("compiled_device")) != _device(result.get("device")):
                raise ValueError("native compiled device differs from selected device")
            if result.get("family") != self._manifest["family"] or result.get("model_id") != self._manifest["model_id"]:
                raise ValueError("native model identity differs from requested pack")
            if self._requested_device != "AUTO":
                if _device(result["device"]) != _device(self._requested_device):
                    raise ValueError("native engine substituted the requested device")
            self.info = {**result, "adapter": self._manifest["adapter"],
                         "requested_device": self._requested_device, "device_source": "native-sdk",
                         "precision": self._manifest["precision"], "stateful_chat": False}
            self._broken = False
        except BaseException as error:
            self._cleanup_after_error(error)
            raise

    @staticmethod
    def _publish(item, frames, stop):
        while not stop.is_set():
            try:
                frames.put_nowait(item)
                return
            except queue.Full:
                stop.wait(0.01)

    def _reader(self, process, frames, stop):
        try:
            while not stop.is_set():
                line = process.stdout.readline(MAX_JSON + 1)
                if not line:
                    raise RuntimeError("native graph process exited: " + "".join(self.stderr)[-3000:])
                if len(line) > MAX_JSON or not line.endswith(b"\n"):
                    raise ValueError("oversized native graph header")
                if line.startswith(b"READY "):
                    self._publish(("READY", _finite_json(line[6:]), b""), frames, stop)
                    continue
                match = re.fullmatch(rb"RESULT ([A-Za-z0-9_-]{1,64}) ([0-9]+) ([0-9]+)\n", line)
                if not match:
                    raise ValueError("invalid native graph protocol line")
                json_size, body_size = int(match[2]), int(match[3])
                body_limit = min(MAX_RESULT, self._manifest["options"].get("max_output_bytes", 32 << 20))
                if not 1 <= json_size <= MAX_JSON or body_size > body_limit:
                    raise ValueError("native graph frame exceeds bounds")
                metadata = _exact(process.stdout, json_size)
                if _exact(process.stdout, 1) != b"\n":
                    raise ValueError("invalid native JSON delimiter")
                body = _exact(process.stdout, body_size)
                if _exact(process.stdout, 1) != b"\n":
                    raise ValueError("invalid native body delimiter")
                self._publish((match[1].decode("ascii"), _finite_json(metadata), body), frames, stop)
        except BaseException as error:
            self._publish(error, frames, stop)

    def _stderr_reader(self, process):
        while True:
            raw = process.stderr.readline(4096)
            if not raw:
                return
            text = raw.decode("utf-8", "replace")
            with self._stderr_lock:
                self.stderr.append(text)
                self._stderr_bytes += len(text.encode("utf-8"))
                while self.stderr and (self._stderr_bytes > MAX_STDERR or len(self.stderr) > 2048):
                    self._stderr_bytes -= len(self.stderr.pop(0).encode("utf-8"))

    @staticmethod
    def _checkpoint(deadline, cancel=None):
        if cancel is not None and cancel():
            from stream_models import StreamCancelled
            raise StreamCancelled("native request cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("native graph request timed out")
        return left

    def _next(self, deadline, cancel=None):
        while True:
            left = self._checkpoint(deadline, cancel)
            try:
                item = self._frames.get(timeout=min(left, 0.05))
            except queue.Empty:
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    @staticmethod
    def _begin_write(owner, data, *, close_input=False):
        done, errors = threading.Event(), []
        # Capturing the owner prevents a late write from reaching a restarted
        # process. Only cleanup closes stdin, after this writer has drained.
        def write():
            try:
                offset = 0
                while offset < len(data):
                    count = owner.process.stdin.write(data[offset:])
                    if type(count) is not int or count <= 0 or count > len(data) - offset:
                        raise OSError("native stdin write did not make progress")
                    offset += count
                owner.process.stdin.flush()
                if close_input:
                    owner.process.stdin.close()
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()
        thread = threading.Thread(target=write, daemon=True)
        owner.threads.append(thread)
        thread.start()
        return done, errors, thread

    def _wait_write(self, write, deadline, cancel=None):
        done, errors, thread = write
        while True:
            left = self._checkpoint(deadline, cancel)
            if done.wait(min(left, 0.01)):
                if errors:
                    raise errors[0]
                # Keep the per-generation thread list bounded over many turns.
                thread.join(timeout=self._checkpoint(deadline, cancel))
                if thread.is_alive():
                    raise TimeoutError("native stdin writer did not drain")
                return

    def _result_identity(self, result):
        if result.get("inference_runtime") != self.info["inference_runtime"]:
            raise ValueError("native result runtime identity differs")
        if _device(result.get("device")) != _device(self.info["device"]):
            raise ValueError("native result substituted its device")
        if (type(result.get("execution_devices_verified")) is not bool or
                result["execution_devices_verified"] != self.info["execution_devices_verified"]):
            raise ValueError("native result execution verification differs")
        for field in ("backend", "family", "model_id"):
            if field in result and result[field] != self.info[field]:
                raise ValueError("native result identity differs: " + field)
            result.setdefault(field, self.info[field])
        if "compiled_device" in result and _device(result["compiled_device"]) != _device(self.info["compiled_device"]):
            raise ValueError("native result compiled device differs")
        result.setdefault("compiled_device", self.info["compiled_device"])

    def _rpc(self, operation, payload, cancel=None, **options):
        if not isinstance(payload, bytes) or not payload or len(payload) > self._manifest["max_input_bytes"] + 16384:
            raise ValueError("native request payload exceeds model bounds")
        if cancel is not None and not callable(cancel):
            predicate = getattr(cancel, "is_set", None)
            if not callable(predicate):
                raise TypeError("cancel must be callable or Event")
            cancel = predicate
        with self._lock:
            if self._closed:
                raise RuntimeError("native model adapter is closed")
            deadline = time.monotonic() + self._request_timeout
            self._checkpoint(deadline, cancel)
            if self._validator is not None:
                self._dispose_validator()
            if self._broken:
                self._dispose(force=True)
            if self._process is None:
                self._start(deadline, cancel)
            try:
                self._counter += 1
                rid = "native" + str(self._counter)
                metadata = json.dumps(dict(operation=operation, **options), ensure_ascii=True, allow_nan=False,
                                      separators=(",", ":")).encode("ascii")
                if len(metadata) > 4096:
                    raise ValueError("native request metadata exceeds bounds")
                request = f"RUN {rid} {len(metadata)} {len(payload)}\n".encode("ascii") + metadata + b"\n" + payload + b"\n"
                owner = self._owner
                writer = self._begin_write(owner, request)
                self._wait_write(writer, deadline, cancel)
                owner.threads.remove(writer[2])
                request_id, result, body = self._next(deadline, cancel)
                if request_id != rid:
                    raise ValueError("native result request ID differs")
                if "error" in result:
                    if body or not isinstance(result["error"], str):
                        raise ValueError("invalid native error result")
                    raise ValueError(result["error"])
                self._result_identity(result)
                if operation == "upscale":
                    if not body or len(body) > self._manifest["options"].get("max_output_bytes", 32 << 20):
                        raise ValueError("native PNG result exceeds model bounds")
                    result["png_base64"] = base64.b64encode(body).decode("ascii")
                elif body or not isinstance(result.get("text"), str):
                    raise ValueError("invalid native text result")
                return result
            except BaseException as error:
                # Cancelling destroys this stateless native request and its SDK
                # state. A later request reloads lazily; no obsolete result escapes.
                self._cleanup_after_error(error)
                raise

    def complete(self, prompt, *, max_new_tokens=None, cancel=None):
        if not isinstance(prompt, str):
            raise ValueError("native prompt must be text")
        options = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        return self._rpc("complete", prompt.encode("utf-8", "strict"), cancel, **options)

    def chat(self, messages, *, max_new_tokens=None, cancel=None):
        payload = json.dumps(messages, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        options = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        return self._rpc("chat", payload, cancel, **options)

    def transcribe(self, wavbytes, *, task="transcribe", language="en", cancel=None):
        result = self._rpc("transcribe", wavbytes, cancel, task=task, language=language)
        from stream_media import validate_wav
        duration = validate_wav(wavbytes, max_audio_seconds=self._manifest["options"].get("max_audio_seconds", 30),
                                max_bytes=self._manifest["max_input_bytes"])["duration"]
        result.update(duration=duration, task=task, language=language,
                      target_language="en" if task == "translate" or language == "en" else None)
        return result

    def upscale(self, pngbytes, *, cancel=None):
        return self._rpc("upscale", pngbytes, cancel)

    def validate_media(self, payload, kind):
        if kind != "camera":
            raise ValueError("native pixel validator requires camera data")
        if not isinstance(payload, bytes) or not payload or len(payload) > self._manifest["max_input_bytes"]:
            raise ValueError("native PNG validation input exceeds bounds")
        args = [str(self._binary), "--validate", "camera", "--max-pixels",
                str(self._manifest["options"].get("max_output_pixels", 5592405)),
                "--max-bytes", str(self._manifest["max_input_bytes"])]
        with self._lock:
            if self._closed:
                raise RuntimeError("native model adapter is closed")
            if self._broken:
                self._dispose(force=True)
            if self._validator is not None:
                self._dispose_validator()
            deadline = time.monotonic() + min(10, self._request_timeout)
            process = subprocess.Popen(args, env=self._env, cwd=ROOT, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            owner = self._validator = _OwnedProcess(process)
            done, output, errors = threading.Event(), bytearray(), []
            def read_output():
                try:
                    while True:
                        chunk = process.stdout.read(min(4096, MAX_JSON + 1 - len(output)))
                        if not chunk:
                            return
                        output.extend(chunk)
                        if len(output) > MAX_JSON:
                            raise ValueError("native PNG validation output exceeds bounds")
                except BaseException as error:
                    errors.append(error)
                finally:
                    done.set()
            owner.threads.extend((threading.Thread(target=read_output, daemon=True),
                                  threading.Thread(target=self._stderr_reader, args=(process,), daemon=True)))
            try:
                for thread in owner.threads:
                    thread.start()
                # Validation reads PNG bytes to EOF, rather than RUN framing.
                self._wait_write(self._begin_write(owner, payload, close_input=True), deadline)
                while not done.wait(min(self._checkpoint(deadline), 0.01)):
                    pass
                if errors:
                    raise errors[0]
                try:
                    returncode = process.wait(timeout=self._checkpoint(deadline))
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError("native PNG validation timed out") from error
                if returncode != 0:
                    raise ValueError("native PNG validation failed")
                result = _finite_json(bytes(output))
                width, height, pixels = (result.get(key) for key in ("width", "height", "output_pixels"))
                limit = self._manifest["options"].get("max_output_pixels", 5592405)
                if (any(type(value) is not int or not 1 <= value <= limit for value in (width, height, pixels)) or
                        pixels != width * height * 16):
                    raise ValueError("native PNG validation geometry exceeds bounds")
                self._dispose_validator()
                return result
            except BaseException as error:
                self._cleanup_after_error(error, validator=True)
                raise

    def _cleanup_after_error(self, error, *, validator=False):
        try:
            if validator:
                self._dispose_validator()
            else:
                self._dispose(force=True)
        except BaseException as cleanup_error:
            # Preserve the cause of the failed operation. The caller still owns
            # this adapter and can retry close; failed startup exposes it too.
            if hasattr(error, "add_note"):
                error.add_note("Native process cleanup is pending: " + str(cleanup_error))
            error.native_adapter = self

    def _drain_owner(self, owner, force):
        process = owner.process
        owner.stop.set()
        deadline = time.monotonic() + self._close_timeout
        shutdown_error = None
        if process.poll() is None and not force:
            try:
                self._wait_write(self._begin_write(owner, b"QUIT\n"), deadline)
                process.wait(timeout=self._checkpoint(deadline))
            except BaseException as error:
                shutdown_error = RuntimeError("native SDK process did not shut down cleanly")
                shutdown_error.__cause__ = error
                force = True
        if process.poll() is None and force:
            # If either call fails, ownership, handles and threads stay intact.
            # Closing BufferedWriter while its writer is blocked would deadlock.
            process.kill()
            deadline = time.monotonic() + self._close_timeout
            process.wait(timeout=max(0, deadline - time.monotonic()))
        if process.poll() is None:
            raise RuntimeError("native process is still running after cleanup")
        for thread in owner.threads:
            if thread is not threading.current_thread() and thread.ident is not None:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in owner.threads):
            raise RuntimeError("native process pipe workers did not drain")
        errors = []
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        return shutdown_error

    def _dispose_validator(self):
        if self._validator is not None:
            self._drain_owner(self._validator, force=True)
            self._validator = None

    def _dispose(self, force=False):
        owner = self._owner
        if owner is None:
            return
        self._broken = True
        shutdown_error = self._drain_owner(owner, force)
        self._process = self._owner = None
        self._broken = False
        if shutdown_error is not None:
            raise shutdown_error

    def close(self):
        with self._lock:
            self._closed = True
            errors = []
            for cleanup in (lambda: self._dispose(force=self._broken), self._dispose_validator):
                try:
                    cleanup()
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise errors[0]

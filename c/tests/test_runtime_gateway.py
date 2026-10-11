"""Real APIServer + GraphEngine integration; only model adapters are fixtures."""
import base64
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import queue
import socket
import struct
import tempfile
import threading
import time
import unittest
import wave
import zlib

from family_registry import default_model_id, family_ids, resolve_model
from openai_server import APIServer
from stream_runtime import GraphCancelled, GraphEngine


def wav_bytes(sample=0):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<h", sample))
    return buffer.getvalue()


def png_bytes(sample=0):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(bytes((0, sample, 2, 3)))) + chunk(b"IEND", b"")


def multipart(fields, boundary="colibri-gateway-fixture"):
    data = bytearray()
    for name, value, filename, extra_headers in fields:
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        data.extend(f"--{boundary}\r\n{disposition}\r\n".encode("ascii"))
        for header in extra_headers:
            data.extend(header.encode("ascii") + b"\r\n")
        data.extend(b"\r\n" + (value if isinstance(value, bytes) else value.encode("utf-8")) + b"\r\n")
    data.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(data), f"multipart/form-data; boundary={boundary}"


class GatewayAdapter:
    info = {"backend": "injected-gateway-fixture", "device": "fixture-CPU",
            "execution_devices_verified": False}

    def __init__(self, block_first=False):
        self.block_first, self.calls, self.closed, self.error = block_first, [], 0, None
        self.entered, self.release, self.lock = queue.Queue(), threading.Event(), threading.Lock()
        self.active, self.cancellation_seen = 0, []

    def _call(self, operation, payload, cancel, options):
        with self.lock:
            assert not self.closed
            index = len(self.calls)
            self.calls.append((operation, payload, options))
            self.active += 1
        self.entered.put(index)
        try:
            if self.block_first and index == 0 and not self.release.wait(5):
                raise RuntimeError("blocked adapter fixture was not released")
            if self.error is not None:
                raise self.error
            cancelled = cancel()
            self.cancellation_seen.append(cancelled)
            if cancelled:
                raise GraphCancelled("fixture cancellation")
            if operation == "transcribe":
                sample = struct.unpack_from("<h", payload, 44)[0]
                return {"text": f"english:{sample}", "segments": [], "language": "en",
                        "target_language": "en", "task": options["task"]}
            if operation == "upscale":
                return {"png_base64": base64.b64encode(payload).decode("ascii"),
                        "width": 4, "height": 4, "scale": 4}
            text = payload if operation == "complete" else payload[-1]["content"]
            return {"text": f"reply:{text}", "stateful_chat": False, "max_new_tokens": options["max_new_tokens"]}
        finally:
            with self.lock:
                self.active -= 1

    def transcribe(self, payload, *, task, language, cancel):
        return self._call("transcribe", payload, cancel, {"task": task, "language": language})

    def upscale(self, payload, *, cancel):
        return self._call("upscale", payload, cancel, {})

    def complete(self, prompt, *, max_new_tokens, cancel):
        return self._call("complete", prompt, cancel, {"max_new_tokens": max_new_tokens})

    def chat(self, messages, *, max_new_tokens, cancel):
        return self._call("chat", messages, cancel, {"max_new_tokens": max_new_tokens})

    def close(self):
        with self.lock:
            assert self.active == 0, "backend destroyed while a gateway operation remained active"
            self.closed += 1


class GatewayTest(unittest.TestCase):
    def setup_gateway(self, family="text_graph", block_first=False, budget=1 << 30, max_sessions=8):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        root = Path(home.name)
        asset = b"bounded gateway adapter fixture"
        (root / "weights.bin").write_bytes(asset)
        adapters = {"speech": "whisper_openvino", "camera_upscale": "super_resolution_openvino",
                    "text_graph": "text_openvino"}
        model_id = "gateway-fixture/" + family
        manifest = {"schema": "colibri.runtime-model/v1", "family": family, "model_id": model_id,
                    "adapter": adapters[family], "files": {"weights.bin": hashlib.sha256(asset).hexdigest()},
                    "precision": "fixture", "max_input_bytes": 4096, "resident_bytes": 256,
                    "workspace_bytes": 128, "session_bytes": 64,
                    "options": {"max_output_bytes": 4096}}
        (root / "colibri_runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
        adapter = GatewayAdapter(block_first)
        engine = GraphEngine(root, memory_budget_bytes=budget, max_sessions=max_sessions,
                             adapter_factory=lambda *unused: adapter)
        server = APIServer(("127.0.0.1", 0), engine, model_id, "fixture-secret", max_tokens=64)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        def cleanup():
            adapter.release.set()
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
            self.assertTrue(engine.close(timeout=2), "gateway backend did not drain during teardown")
            self.assertEqual(adapter.closed, 1)
        self.addCleanup(cleanup)
        self.server, self.engine, self.adapter, self.root, self.model_id = server, engine, adapter, root, model_id
        return engine, adapter

    def request(self, method, path, body=None, *, raw=None, content_type="application/json", auth=True, headers=None):
        if raw is None and body is not None:
            body = {"model": self.model_id, **body}
            raw = json.dumps(body).encode("utf-8")
        request_headers = {"Content-Type": content_type}
        if auth:
            request_headers["Authorization"] = "Bearer fixture-secret"
        if headers:
            request_headers.update(headers)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=raw, headers=request_headers)
            response = connection.getresponse()
            data = response.read()
            result = json.loads(data) if data else None
            return response.status, result, {key.lower(): value for key, value in response.getheaders()}
        finally:
            connection.close()

    def post(self, path, body=None, **kwargs):
        return self.request("POST", path, {} if body is None else body, **kwargs)

    def assert_error(self, status, response, expected):
        self.assertEqual(status, expected, response)
        self.assertIsInstance(response["error"]["message"], str)
        self.assertTrue(response["error"]["message"])

    def events_until(self, sid, terminal="finished", initial=()):
        result, deadline = list(initial), time.monotonic() + 3
        while not any(event["type"] == terminal for event in result):
            self.assertLess(time.monotonic(), deadline, "runtime gateway did not deliver terminal event")
            status, response, _ = self.request("GET", f"/v1/runtime/streams/{sid}/events?wait_ms=100")
            self.assertEqual(status, 200, response)
            result.extend(response["events"])
        return result

    def test_registry_resolution_and_served_capabilities(self):
        self.setup_gateway()
        resolved = resolve_model(self.root)
        self.assertEqual(resolved.descriptor.runtime_kind, "graph")
        self.assertEqual(resolved.descriptor.id, "text_graph")
        self.assertEqual(default_model_id(resolved), self.model_id)
        self.assertNotIn("text_graph", family_ids())
        self.assertIn("text_graph", family_ids(include_graph=True))
        status, response, headers = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        entry = response["data"][0]
        self.assertEqual(entry["id"], self.model_id)
        self.assertEqual(entry["input_modalities"], ["text"])
        self.assertEqual(entry["capabilities"], ["chat", "completion"])
        self.assertEqual(entry["runtime"]["max_input_bytes"], 4096)
        self.assertEqual(entry["runtime"]["compiled_device"], "fixture-CPU")
        self.assertFalse(entry["runtime"]["execution_devices_verified"])
        self.assertIsNone(entry["runtime"]["executed_device"])
        self.assertTrue(headers["x-request-id"].startswith("req_"))

    def test_registry_metadata_does_not_impose_default_one_gib_budget(self):
        self.setup_gateway()
        path = self.root / "colibri_runtime.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["resident_bytes"] = 2 << 30
        path.write_text(json.dumps(manifest), encoding="utf-8")
        resolved = resolve_model(self.root)
        self.assertEqual(resolved.descriptor.id, "text_graph")
        self.assertEqual(default_model_id(resolved), self.model_id)
        self.assertEqual(resolved.family_config["resident_bytes"], 2 << 30)
        with self.assertRaises(MemoryError):
            GraphEngine(self.root, adapter_factory=lambda *unused: self.adapter)

    def test_auth_host_and_public_health(self):
        self.setup_gateway()
        status, body, _ = self.request("GET", "/v1/models", auth=False)
        self.assert_error(status, body, 401)
        status, body, _ = self.post("/v1/completions", {"prompt": "private"}, auth=False)
        self.assert_error(status, body, 401)
        self.assertEqual(self.adapter.calls, [])
        status, body, _ = self.request("GET", "/health", auth=False)
        self.assertEqual((status, body), (200, {"status": "ok"}))
        status, body, _ = self.request("GET", "/health", headers={"Host": "attacker.invalid"})
        self.assert_error(status, body, 403)
        status, body, _ = self.post("/v1/completions", {"prompt": "private"}, headers={"Host": "attacker.invalid"})
        self.assert_error(status, body, 403)
        self.assertEqual(self.adapter.calls, [])
        status, body, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["runtime"]["model_id"], self.model_id)

    def test_unauthorized_body_drain_keeps_next_graph_request_framed(self):
        self.setup_gateway()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        self.addCleanup(connection.close)
        payload = json.dumps({"model": self.model_id, "prompt": "ignored"}).encode()
        connection.request("POST", "/v1/completions", payload, {"Content-Type": "application/json"})
        denied = connection.getresponse()
        self.assertEqual(denied.status, 401)
        denied.read()
        payload = json.dumps({"model": self.model_id, "prompt": "next"}).encode()
        connection.request("POST", "/v1/completions", payload,
                           {"Content-Type": "application/json", "Authorization": "Bearer fixture-secret"})
        allowed = connection.getresponse()
        self.assertEqual(allowed.status, 200)
        self.assertEqual(json.loads(allowed.read())["choices"][0]["text"], "reply:next")
        self.assertEqual(len(self.adapter.calls), 1)

    def test_standard_text_routes_and_unsupported_options(self):
        self.setup_gateway()
        status, response, _ = self.post("/v1/completions", {"prompt": "hello", "max_tokens": 16})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["choices"][0]["text"], "reply:hello")
        status, response, _ = self.post("/v1/chat/completions", {
            "messages": [{"role": "system", "content": "English"}, {"role": "user", "content": "hello"}]})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"], {"role": "assistant", "content": "reply:hello"})
        self.assertFalse(response["runtime"]["stateful_chat"])
        calls = len(self.adapter.calls)
        bad_bodies = ({"prompt": "hello", "stream": True}, {"prompt": "hello", "stream": 0},
                      {"prompt": "hello", "temperature": 0}, {"prompt": "hello", "max_tokens": True},
                      {"prompt": "hello", "max_tokens": 65}, {"prompt": "hello", "max_tokens": 0},
                      {"prompt": "hello", "messages": [{"role": "user", "content": "ignored"}]})
        for body in bad_bodies:
            with self.subTest(body=body):
                status, response, _ = self.post("/v1/completions", body)
                self.assert_error(status, response, 400)
        status, response, _ = self.post("/v1/chat/completions", {"messages": [{"role": "tool", "content": "x"}]})
        self.assert_error(status, response, 400)
        status, response, _ = self.post("/v1/completions", {"model": "wrong", "prompt": "hello"})
        self.assert_error(status, response, 404)
        self.assertEqual(len(self.adapter.calls), calls)

    def test_strict_json_and_content_type_are_rejected_before_inference(self):
        self.setup_gateway()
        model = json.dumps(self.model_id)
        invalid = [f'{{"model":{model},"prompt":"one","prompt":"two"}}'.encode(),
                   f'{{"model":{model},"prompt":"x","max_tokens":NaN}}'.encode(),
                   f'{{"model":{model},"prompt":"x","max_tokens":1e9999}}'.encode(),
                   f'{{"model":{model},"prompt":"\\ud800"}}'.encode(), b"[]", b'{"prompt":"\xff"}', b"{"]
        for payload in invalid:
            with self.subTest(payload=payload):
                status, response, _ = self.post("/v1/completions", raw=payload)
                self.assert_error(status, response, 400)
        status, response, _ = self.post("/v1/completions", {"prompt": "hello"}, content_type="text/plain")
        self.assert_error(status, response, 415)
        self.assertEqual(self.adapter.calls, [])

    def test_ambiguous_and_truncated_http_framing(self):
        self.setup_gateway()
        for extra, short in (("Content-Length: 2\r\nContent-Length: 2\r\n", False),
                             ("Content-Length: 2\r\nTransfer-Encoding: chunked\r\n", False),
                             ("Content-Length: 7\r\n", True)):
            with self.subTest(extra=extra):
                request = ("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                           "Authorization: Bearer fixture-secret\r\nContent-Type: application/json\r\n"
                           "Connection: close\r\n" + extra + "\r\n{}").encode()
                with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as sock:
                    sock.sendall(request)
                    if short:
                        sock.shutdown(socket.SHUT_WR)
                    response = http.client.HTTPResponse(sock)
                    response.begin()
                    self.assert_error(response.status, json.loads(response.read()), 400)
        self.assertEqual(self.adapter.calls, [])

    def test_audio_json_translation_and_standard_multipart(self):
        self.setup_gateway("speech")
        payload = wav_bytes(7)
        status, response, _ = self.post("/v1/audio/translations", {
            "audio_base64": base64.b64encode(payload).decode(), "language": None})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["text"], "english:7")
        self.assertEqual(response["target_language"], "en")
        self.assertEqual(self.adapter.calls[-1][2], {"task": "translate", "language": None})
        raw, content_type = multipart([("model", self.model_id, None, []), ("language", "en", None, []),
                                       ("file", payload, "chunk.wav", ["Content-Type: audio/wav"])])
        status, response, _ = self.post("/v1/audio/transcriptions", raw=raw, content_type=content_type)
        self.assertEqual(status, 200, response)
        self.assertEqual(response["task"], "transcribe")
        self.assertEqual(self.adapter.calls[-1][1], payload)

    def test_multipart_duplicates_encoding_and_boundary_errors(self):
        self.setup_gateway("speech")
        model = ("model", self.model_id, None, [])
        audio = ("file", wav_bytes(0), "chunk.wav", [])
        cases = [[model, audio, ("file", wav_bytes(1), "second.wav", [])],
                 [model, model, audio], [model, audio, ("unknown", "value", None, [])],
                 [model, ("file", wav_bytes(0), "chunk.wav", ["Content-Transfer-Encoding: base64"])],
                 [model], [model, audio, ("language", b"\xff", None, [])]]
        for fields in cases:
            with self.subTest(fields=[entry[0] for entry in fields]):
                raw, content_type = multipart(fields)
                status, response, _ = self.post("/v1/audio/transcriptions", raw=raw, content_type=content_type)
                self.assert_error(status, response, 400)
        raw, content_type = multipart([model, audio])
        raw = raw.rsplit(b"--colibri-gateway-fixture--\r\n", 1)[0]
        status, response, _ = self.post("/v1/audio/transcriptions", raw=raw, content_type=content_type)
        self.assert_error(status, response, 400)
        self.assertEqual(self.adapter.calls, [])

    def test_multipart_content_type_and_parameter_aliases(self):
        self.setup_gateway("speech")
        raw, content_type = multipart([("model", self.model_id, None, []),
                                       ("file", wav_bytes(), "chunk.wav", [])])
        altered = raw.replace(b'name="file";', b'name="file"; name="model";')
        cases = [(raw, content_type.replace("multipart/form-data", "multipart/form-datax"), 415),
                 (raw, content_type + "; boundary=second-boundary", 400),
                 (altered, content_type, 400), (raw, "multipart/form-data; boundary=\u00e9", 400)]
        for payload, kind, expected in cases:
            with self.subTest(content_type=kind, duplicate_name=payload is altered):
                status, response, _ = self.post("/v1/audio/transcriptions", raw=payload, content_type=kind)
                self.assert_error(status, response, expected)
        self.assertEqual(self.adapter.calls, [])

    def test_bounded_media_wrong_modality_and_unsupported_audio_fields(self):
        self.setup_gateway("speech")
        valid = base64.b64encode(wav_bytes()).decode()
        for body in ({"audio_base64": "!"}, {"audio_base64": ""},
                     {"audio_base64": base64.b64encode(b"x" * 4097).decode()},
                     {"audio_base64": base64.b64encode(b"bad WAV").decode()},
                     {"audio_base64": valid, "response_format": "text"},
                     {"audio_base64": valid, "language": "fr"},
                     {"audio_base64": valid, "temperature": 0.5}):
            status, response, _ = self.post("/v1/audio/transcriptions", body)
            self.assert_error(status, response, 400)
        status, response, _ = self.post("/v1/camera/upscale", {"png_base64": base64.b64encode(png_bytes()).decode()})
        self.assert_error(status, response, 400)
        self.assertEqual(self.adapter.calls, [])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "camera validation requires optional Pillow")
    def test_camera_upscale_and_latest_pending_frame_events(self):
        self.setup_gateway("camera_upscale", block_first=True)
        status, response, _ = self.post("/v1/runtime/streams", {"kind": "camera", "queue_capacity": 1})
        self.assertEqual(status, 201, response)
        sid = response["stream_id"]
        receipts = []
        for sequence in range(3):
            status, receipt, _ = self.post(f"/v1/runtime/streams/{sid}/push", {
                "sequence": sequence, "timestamp_ms": sequence * 10,
                "payload_base64": base64.b64encode(png_bytes(sequence)).decode(), "final": sequence == 2})
            self.assertEqual(status, 200, receipt)
            receipts.append(receipt)
            if sequence == 0:
                self.adapter.entered.get(timeout=1)
        self.assertEqual(receipts[2]["dropped"]["request_id"], receipts[1]["request_id"])
        self.adapter.release.set()
        events = self.events_until(sid)
        self.assertEqual([event["sequence"] for event in events if event["type"] == "result"], [0, 2])
        dropped = [event for event in events if event["type"] == "dropped"]
        self.assertEqual(dropped[0]["request_id"], receipts[1]["request_id"])
        status, response, _ = self.post("/v1/camera/upscale", {
            "png_base64": base64.b64encode(png_bytes(0)).decode(), "scale": 2})
        self.assert_error(status, response, 400)
        status, response, _ = self.post("/v1/camera/upscale", {"png_base64": base64.b64encode(png_bytes(0)).decode()})
        self.assertEqual(status, 200, response)
        self.assertEqual((response["width"], response["height"], response["scale"]), (4, 4, 4))

    def test_audio_stream_backpressure_retry_events_and_close(self):
        self.setup_gateway("speech", block_first=True)
        status, created, _ = self.post("/v1/runtime/streams", {"kind": "audio", "queue_capacity": 1})
        self.assertEqual(status, 201, created)
        sid = created["stream_id"]
        def push(sequence, final=False):
            return self.post(f"/v1/runtime/streams/{sid}/push", {"sequence": sequence,
                "timestamp_ms": sequence * 1000, "payload_base64": base64.b64encode(wav_bytes(sequence)).decode(),
                "final": final})
        status, first, _ = push(0)
        self.assertEqual(status, 200)
        self.adapter.entered.get(timeout=1)
        self.assertEqual(push(1)[0], 200)
        status, response, _ = push(2, final=True)
        self.assert_error(status, response, 429)
        self.adapter.release.set()
        self.adapter.entered.get(timeout=1)
        self.assertEqual(push(2, final=True)[0], 200)
        events = self.events_until(sid)
        results = [event for event in events if event["type"] == "result"]
        self.assertEqual([event["sequence"] for event in results], [0, 1, 2])
        self.assertEqual(results[0]["request_id"], first["request_id"])
        self.assertEqual([event["result"]["text"] for event in results], ["english:0", "english:1", "english:2"])
        status, response, _ = self.post(f"/v1/runtime/streams/{sid}/close")
        self.assertEqual((status, response["closed"]), (200, True))
        status, response, _ = self.request("GET", f"/v1/runtime/streams/{sid}/events")
        self.assert_error(status, response, 404)

    def test_stream_cancel_close_fence_unknown_ids_and_query_validation(self):
        self.setup_gateway("speech", block_first=True)
        status, created, _ = self.post("/v1/runtime/streams", {"kind": "audio"})
        self.assertEqual(status, 201)
        sid = created["stream_id"]
        self.post(f"/v1/runtime/streams/{sid}/push", {"sequence": 0, "timestamp_ms": 0,
                    "payload_base64": base64.b64encode(wav_bytes()).decode()})
        self.adapter.entered.get(timeout=1)
        for suffix in ("?max_events=65", "?wait_ms=1001", "?max_events=", "?wait_ms=nan",
                       "?wait_ms=0&wait_ms=1", "?ignored=yes"):
            status, response, _ = self.request("GET", f"/v1/runtime/streams/{sid}/events{suffix}")
            self.assert_error(status, response, 400)
        status, response, _ = self.post(f"/v1/runtime/streams/{sid}/cancel")
        self.assertEqual((status, response["cancelled"]), (200, True))
        events = self.events_until(sid, terminal="cancelled")
        self.assertEqual([event["type"] for event in events], ["cancelled"])
        status, response, _ = self.post(f"/v1/runtime/streams/{sid}/close")
        self.assertEqual((status, response["closed"]), (200, False))
        self.assertEqual(self.adapter.closed, 0)
        self.adapter.release.set()
        status, response, _ = self.post(f"/v1/runtime/streams/{sid}/close")
        self.assertEqual((status, response["closed"]), (200, True))
        for action in ("cancel", "close"):
            status, response, _ = self.post(f"/v1/runtime/streams/unknown/{action}")
            self.assert_error(status, response, 404)

    def test_stream_validation_session_limits_and_memory_mapping(self):
        self.setup_gateway("speech", max_sessions=1)
        for body in ({"kind": "audio", "queue_capacity": True}, {"kind": "camera"},
                     {"kind": "audio", "options": {"language": "fr"}}, {"kind": "audio", "ignored": True}):
            status, response, _ = self.post("/v1/runtime/streams", body)
            self.assert_error(status, response, 400)
        status, created, _ = self.post("/v1/runtime/streams", {"kind": "audio"})
        self.assertEqual(status, 201)
        status, response, _ = self.post("/v1/runtime/streams", {"kind": "audio"})
        self.assert_error(status, response, 429)
        status, response, _ = self.post(f"/v1/runtime/streams/{created['stream_id']}/push", {
            "sequence": True, "timestamp_ms": 0, "payload_base64": base64.b64encode(wav_bytes()).decode()})
        self.assert_error(status, response, 400)
        # A separate admitted model with no remaining session-estimate budget.
        self.setup_gateway("speech", budget=384)
        status, response, _ = self.post("/v1/runtime/streams", {"kind": "audio"})
        self.assert_error(status, response, 429)

    def test_backend_cancel_failure_and_memory_errors_map_without_losing_server(self):
        self.setup_gateway()
        for exception, expected in ((GraphCancelled("fixture cancellation"), 409),
                                    (RuntimeError("fixture backend failure"), 503),
                                    (MemoryError("fixture memory failure"), 429)):
            self.adapter.error = exception
            status, response, _ = self.post("/v1/completions", {"prompt": "hello"})
            self.assert_error(status, response, expected)
        self.adapter.error = None
        status, response, _ = self.post("/v1/completions", {"prompt": "recovered"})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["choices"][0]["text"], "reply:recovered")

    def test_client_disconnect_cancels_live_request_then_backend_recovers(self):
        self.setup_gateway(block_first=True)
        payload = json.dumps({"model": self.model_id, "prompt": "abandoned"}).encode()
        request = ("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                   "Authorization: Bearer fixture-secret\r\nContent-Type: application/json\r\n"
                   f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload
        sock = socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3)
        try:
            sock.sendall(request)
            self.adapter.entered.get(timeout=1)
        finally:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        self.adapter.release.set()
        status, response, _ = self.post("/v1/completions", {"prompt": "recovered"})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["choices"][0]["text"], "reply:recovered")
        self.assertEqual(self.adapter.cancellation_seen, [True, False])


if __name__ == "__main__":
    unittest.main()

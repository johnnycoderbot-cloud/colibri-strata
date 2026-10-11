"""Behavioral stream/runtime tests; adapters are injected fixtures only."""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave
import zlib

from stream_runtime import GraphCancelled, GraphEngine, ManifestError, read_manifest_metadata, validate_manifest


def wav_frame(value=0):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<h", value))
    return buffer.getvalue()


def png_frame(value=0):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(bytes((0, value, 2, 3)))) + chunk(b"IEND", b"")


class FakeAdapter:
    info = {"backend": "injected-test-fixture", "device": "fixture-CPU"}

    def __init__(self, block_first=False):
        self.block_first = block_first
        self.release = threading.Event()
        self.entered = queue.Queue()
        self.lock = threading.Lock()
        self.calls, self.active, self.max_active, self.closed = [], 0, 0, 0
        self.result_override, self.callback, self.close_callback, self.cancel_observations = None, None, None, []

    def _operation(self, payload, cancel, **options):
        with self.lock:
            assert not self.closed, "backend used after destruction"
            index = len(self.calls)
            self.calls.append((payload, options))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.put(index)
        try:
            self.cancel_observations.append(cancel())
            if self.block_first and index == 0:
                if not self.release.wait(5):
                    raise RuntimeError("fixture operation was not released")
            if self.callback:
                self.callback()
            self.cancel_observations.append(cancel())
            if self.result_override is not None:
                return self.result_override
            if options.get("operation") in ("complete", "chat"):
                text = payload if isinstance(payload, str) else payload[-1]["content"]
                return {"text": "reply:" + text, "stateful_chat": False,
                        "max_new_tokens": options["max_new_tokens"] or 8}
            if options:
                value = struct.unpack_from("<h", payload, 44)[0]
                return {"text": str(value), "segments": [{"text": str(value), "start": 0.1, "end": 0.2}],
                        "language": "en", "target_language": "en", "task": options["task"]}
            return {"png_base64": base64.b64encode(payload).decode("ascii"),
                    "width": 4, "height": 4, "scale": 4}
        finally:
            with self.lock:
                self.active -= 1

    def transcribe(self, payload, *, task, language, cancel):
        return self._operation(payload, cancel, task=task, language=language)

    def upscale(self, payload, *, cancel):
        return self._operation(payload, cancel)

    def complete(self, prompt, *, max_new_tokens=None, cancel):
        return self._operation(prompt, cancel, operation="complete", max_new_tokens=max_new_tokens)

    def chat(self, messages, *, max_new_tokens=None, cancel):
        return self._operation(messages, cancel, operation="chat", max_new_tokens=max_new_tokens)

    def close(self):
        with self.lock:
            assert self.active == 0, "backend freed while inference remained active"
            self.closed += 1
        if self.close_callback:
            self.close_callback()


class RuntimeFixture(unittest.TestCase):
    def model(self, family="speech"):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        root = Path(home.name)
        asset = b"declared test fixture asset"
        (root / "weights.bin").write_bytes(asset)
        adapters = {"speech": "whisper_openvino", "camera_upscale": "super_resolution_openvino",
                    "text_graph": "text_openvino"}
        manifest = {"schema": "colibri.runtime-model/v1", "family": family,
                    "model_id": "fixture/" + family, "adapter": adapters[family],
                    "files": {"weights.bin": hashlib.sha256(asset).hexdigest()}, "precision": "F32 fixture",
                    "max_input_bytes": 4096, "resident_bytes": 256, "workspace_bytes": 128,
                    "session_bytes": 64, "options": {"max_output_bytes": 4096},
                    "provenance": {"source": "test fixture", "license": "test"}}
        self.write_manifest(root, manifest)
        return root, manifest

    @staticmethod
    def write_manifest(root, manifest):
        (root / "colibri_runtime.json").write_text(json.dumps(manifest), encoding="utf-8")

    def engine(self, family="speech", block_first=False, **kwargs):
        root, manifest = self.model(family)
        adapter = FakeAdapter(block_first)
        engine = GraphEngine(root, adapter_factory=lambda *unused: adapter, **kwargs)
        self.addCleanup(engine.close)
        self.addCleanup(adapter.release.set)
        return engine, adapter, root, manifest

    def collect_finished(self, engine, sid, initial=()):
        events, deadline = list(initial), time.monotonic() + 3
        while not any(event["type"] == "finished" for event in events):
            self.assertLess(time.monotonic(), deadline, "stream did not deliver its final event")
            events.extend(engine.stream_events(sid, wait_ms=100))
        return events


class RuntimeTest(RuntimeFixture):
    def test_manifest_and_actual_backend_info(self):
        engine, adapter, root, manifest = self.engine()
        admitted, path, assets, base = validate_manifest(root)
        self.assertEqual(admitted, manifest)
        self.assertEqual(path, root.resolve())
        self.assertEqual(assets, (root / "weights.bin").stat().st_size)
        self.assertEqual(base, 384)
        self.assertEqual(engine.info()["backend"], adapter.info["backend"])
        self.assertEqual(engine.info()["compiled_device"], adapter.info["device"])
        self.assertEqual(engine.info()["requested_device"], "AUTO")
        self.assertFalse(engine.info()["execution_devices_verified"])
        self.assertIsNone(engine.info()["executed_device"])
        self.assertTrue(engine.info()["memory_estimates_only"])
        self.assertEqual(engine.info()["max_input_bytes"], manifest["max_input_bytes"])
        self.assertEqual(engine.info()["max_output_bytes"], manifest["options"]["max_output_bytes"])
        changed = engine.info()
        changed["backend"] = "invented"
        self.assertEqual(engine.info()["backend"], adapter.info["backend"])
        self.assertEqual(engine.runtime_kind, "graph")
        self.assertEqual(engine.modality, "audio")
        self.assertEqual(engine.transcribe(wav_frame(7))["text"], "7")
        self.assertEqual(adapter.calls[0][1], {"task": "transcribe", "language": "en"})
        self.assertEqual(engine.transcribe(wav_frame(7), task="translate", language=None)["target_language"], "en")
        for task, language in (("summarize", "en"), ("transcribe", "es"), (True, "en")):
            with self.assertRaises(ValueError):
                engine.transcribe(wav_frame(), task=task, language=language)

    def test_audio_backpressure_retries_order_and_final(self):
        engine, adapter, _, _ = self.engine(block_first=True)
        sid = engine.create_stream("audio", queue_capacity=1)
        first = engine.push_stream(sid, 0, 0, wav_frame(0))
        self.assertEqual(adapter.entered.get(timeout=1), 0)
        second = engine.push_stream(sid, 1, 1250, wav_frame(1))
        with self.assertRaises(BufferError):
            engine.push_stream(sid, 2, 2500, wav_frame(2), final=True)
        adapter.release.set()
        self.assertEqual(adapter.entered.get(timeout=1), 1)
        final = engine.push_stream(sid, 2, 2500, wav_frame(2), final=True)
        with self.assertRaises(RuntimeError):
            engine.push_stream(sid, 3, 3500, wav_frame(3))
        events = self.collect_finished(engine, sid)
        results = [event for event in events if event["type"] == "result"]
        self.assertEqual([event["sequence"] for event in results], [0, 1, 2])
        self.assertEqual([event["request_id"] for event in results],
                         [first["request_id"], second["request_id"], final["request_id"]])
        self.assertEqual([event["result"]["text"] for event in results], ["0", "1", "2"])
        self.assertAlmostEqual(results[1]["result"]["segments"][0]["start"], 1.35)
        self.assertEqual(events[-1]["type"], "finished")
        self.assertEqual(events[-1]["request_id"], final["request_id"])
        self.assertFalse(any(event["type"] == "dropped" for event in events))

    def test_output_backpressure_preserves_all_accepted_audio(self):
        engine, adapter, _, _ = self.engine()
        sid = engine.create_stream("audio", queue_capacity=1)
        for sequence in range(4):
            engine.push_stream(sid, sequence, sequence * 10, wav_frame(sequence))
            self.assertEqual(adapter.entered.get(timeout=1), sequence)
        engine.push_stream(sid, 4, 40, wav_frame(4))
        with self.assertRaises(BufferError):
            engine.push_stream(sid, 5, 50, wav_frame(5), final=True)
        initial = engine.stream_events(sid)
        self.assertEqual([e["sequence"] for e in initial], [0, 1, 2])
        self.assertEqual(adapter.entered.get(timeout=1), 4)
        engine.push_stream(sid, 5, 50, wav_frame(5), final=True)
        events = self.collect_finished(engine, sid, initial)
        self.assertEqual([e["sequence"] for e in events if e["type"] == "result"], list(range(6)))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional PNG decoder unavailable")
    def test_camera_drops_old_pending_identity_only(self):
        engine, adapter, _, _ = self.engine("camera_upscale", block_first=True)
        sid = engine.create_stream("camera", queue_capacity=2)
        frames = [png_frame(i) for i in range(4)]
        first = engine.push_stream(sid, 0, 0, frames[0])
        self.assertEqual(adapter.entered.get(timeout=1), 0)
        old = engine.push_stream(sid, 1, 10, frames[1])
        engine.push_stream(sid, 2, 20, frames[2])
        newest = engine.push_stream(sid, 3, 30, frames[3], final=True)
        self.assertEqual(newest["dropped"]["sequence"], 1)
        self.assertEqual(newest["dropped"]["request_id"], old["request_id"])
        dropped = engine.stream_events(sid)
        self.assertEqual(dropped[0]["type"], "dropped")
        self.assertEqual(dropped[0]["request_id"], old["request_id"])
        adapter.release.set()
        events = self.collect_finished(engine, sid, dropped)
        results = [e for e in events if e["type"] == "result"]
        self.assertEqual([e["sequence"] for e in results], [0, 2, 3])
        self.assertEqual(results[0]["request_id"], first["request_id"])
        self.assertEqual([call[0] for call in adapter.calls], [frames[0], frames[2], frames[3]])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional PNG decoder unavailable")
    def test_camera_event_bound_refuses_unreportable_drop_and_can_retry(self):
        engine, adapter, _, _ = self.engine("camera_upscale", block_first=True)
        sid = engine.create_stream("camera", queue_capacity=1)
        engine.push_stream(sid, 0, 0, png_frame(0))
        adapter.entered.get(timeout=1)
        for sequence in range(1, 5):
            receipt = engine.push_stream(sid, sequence, sequence, png_frame(sequence))
        with self.assertRaises(BufferError):
            engine.push_stream(sid, 5, 5, png_frame(5))
        events = engine.stream_events(sid)
        self.assertEqual([e["sequence"] for e in events], [1, 2, 3])
        retry = engine.push_stream(sid, 5, 5, png_frame(5), final=True)
        self.assertEqual(retry["dropped"]["request_id"], receipt["request_id"])
        self.assertEqual(retry["dropped_total"], 4)

    def test_cancel_inflight_discards_results_and_pending_frames(self):
        engine, adapter, _, _ = self.engine(block_first=True)
        sid = engine.create_stream("audio")
        active = engine.push_stream(sid, 0, 0, wav_frame(0))
        adapter.entered.get(timeout=1)
        pending = engine.push_stream(sid, 1, 1, wav_frame(1), final=True)
        cancelled = engine.cancel_stream(sid)
        self.assertEqual(cancelled["in_flight"]["request_id"], active["request_id"])
        self.assertEqual(cancelled["discarded"][0]["request_id"], pending["request_id"])
        self.assertGreater(cancelled["generation"], active["generation"])
        self.assertEqual([e["type"] for e in engine.stream_events(sid)], ["cancelled"])
        adapter.release.set()
        self.assertTrue(engine.close_stream(sid, timeout=1))
        self.assertEqual(len(adapter.calls), 1)
        self.assertIn(True, adapter.cancel_observations)
        with self.assertRaises(KeyError):
            engine.stream_events(sid)

    def test_close_timeout_retains_backend_and_sessions_until_drained(self):
        engine, adapter, _, _ = self.engine(block_first=True)
        sid = engine.create_stream("audio")
        engine.push_stream(sid, 0, 0, wav_frame())
        adapter.entered.get(timeout=1)
        self.assertFalse(engine.close(timeout=0.01))
        self.assertEqual(adapter.closed, 0)
        self.assertEqual(engine.info()["active_sessions"], 1)
        self.assertTrue(engine.info()["closing"])
        self.assertFalse(engine.info()["closed"])
        with self.assertRaises(RuntimeError):
            engine.create_stream("audio")
        adapter.release.set()
        self.assertTrue(engine.close(timeout=1))
        self.assertTrue(engine.close(timeout=1))
        self.assertEqual(adapter.closed, 1)
        self.assertEqual(engine.info()["session_estimated_bytes"], 0)

    def test_backend_serialization_and_direct_close_drain(self):
        engine, adapter, _, _ = self.engine(block_first=True)
        results, errors = [], []
        def request(value):
            try:
                results.append(engine.transcribe(wav_frame(value)))
            except Exception as error:
                errors.append(error)
        first = threading.Thread(target=request, args=(0,))
        second = threading.Thread(target=request, args=(1,))
        first.start()
        adapter.entered.get(timeout=1)
        second.start()
        with self.assertRaises(queue.Empty):
            adapter.entered.get(timeout=0.03)
        self.assertFalse(engine.close(timeout=0.01))
        self.assertEqual(adapter.closed, 0)
        adapter.release.set()
        first.join(1); second.join(1)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(adapter.max_active, 1)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(isinstance(error, GraphCancelled) for error in errors))
        self.assertEqual(results, [])
        self.assertTrue(engine.close(timeout=1))

    def test_close_called_inside_backend_defers_destruction(self):
        engine, adapter, _, _ = self.engine()
        observed = []
        adapter.callback = lambda: observed.append(engine.close(timeout=0))
        with self.assertRaises(GraphCancelled):
            engine.transcribe(wav_frame())
        self.assertEqual(observed, [False])
        self.assertEqual(adapter.closed, 0)
        self.assertTrue(engine.close(timeout=1))
        self.assertEqual(adapter.closed, 1)

    def test_concurrent_close_destroys_backend_once(self):
        engine, adapter, _, _ = self.engine(block_first=True)
        sid = engine.create_stream("audio")
        engine.push_stream(sid, 0, 0, wav_frame())
        adapter.entered.get(timeout=1)
        outcomes, failures = [], []
        def closer():
            try:
                outcomes.append(engine.close(timeout=1))
            except Exception as error:
                failures.append(error)
        threads = [threading.Thread(target=closer) for _ in range(2)]
        for thread in threads: thread.start()
        adapter.release.set()
        for thread in threads: thread.join(1)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(outcomes, [True, True])
        self.assertEqual(failures, [])
        self.assertEqual(adapter.closed, 1)

    def test_backend_close_callback_cannot_recursively_destroy_itself(self):
        engine, adapter, _, _ = self.engine()
        observed = []
        adapter.close_callback = lambda: observed.append(engine.close(timeout=0))
        self.assertTrue(engine.close(timeout=1))
        self.assertEqual(observed, [False])
        self.assertEqual(adapter.closed, 1)

    def test_session_limits_and_estimate_budget(self):
        engine, _, _, _ = self.engine(max_sessions=1)
        sid = engine.create_stream("audio")
        with self.assertRaises(BufferError):
            engine.create_stream("audio")
        reservation = engine.info()["session_estimated_bytes"]
        self.assertGreater(reservation, 0)
        self.assertTrue(engine.close_stream(sid, timeout=1))
        self.assertEqual(engine.info()["session_estimated_bytes"], 0)
        new = engine.create_stream("audio")
        self.assertNotEqual(sid, new)
        limited, _, _, _ = self.engine(memory_budget_bytes=384)
        with self.assertRaises(MemoryError):
            limited.create_stream("audio")
        self.assertEqual(limited.info()["active_sessions"], 0)
        self.assertEqual(limited.info()["session_estimated_bytes"], 0)

    def test_invalid_input_does_not_consume_sequence(self):
        engine, adapter, _, _ = self.engine(block_first=True)
        sid = engine.create_stream("audio")
        for sequence, timestamp in ((True, 0), (0, True), (0, float("nan")),
                                    (0, float("inf")), (0, 10**1000), (-1, 0)):
            with self.assertRaises(ValueError):
                engine.push_stream(sid, sequence, timestamp, wav_frame())
        for payload in (b"not WAV", b"", bytearray(wav_frame()), b"x" * 4097):
            with self.assertRaises(ValueError):
                engine.push_stream(sid, 0, 0, payload)
        with self.assertRaises(ValueError):
            engine.push_stream(sid, 0, 0, wav_frame(), final=1)
        engine.push_stream(sid, 0, 0, wav_frame())
        adapter.entered.get(timeout=1)
        for sequence, timestamp in ((0, 0), (1, -1)):
            with self.assertRaises(ValueError):
                engine.push_stream(sid, sequence, timestamp, wav_frame())
        engine.push_stream(sid, 1, 0, wav_frame(1), final=True)
        adapter.release.set()
        events = self.collect_finished(engine, sid)
        self.assertEqual([e["sequence"] for e in events if e["type"] == "result"], [0, 1])
        for options in ({"task": "summarize"}, {"language": "fr"}, {"other": True}, []):
            with self.assertRaises(ValueError):
                engine.create_stream("audio", options=options)
        for capacity in (0, True, 17, float("nan")):
            with self.assertRaises(ValueError):
                engine.create_stream("audio", queue_capacity=capacity)
        with self.assertRaises(ValueError):
            engine.create_stream("camera")
        with self.assertRaises(ValueError):
            engine.stream_events(sid, max_events=True)
        with self.assertRaises(ValueError):
            engine.stream_events(sid, wait_ms=float("nan"))

    def test_bad_or_excessive_adapter_results_become_stable_errors(self):
        engine, adapter, _, _ = self.engine()
        adapter.result_override = {"text": "x" * 4097, "segments": []}
        sid = engine.create_stream("audio")
        receipt = engine.push_stream(sid, 4, 100, wav_frame(), final=True)
        events = self.collect_finished(engine, sid)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["request_id"], receipt["request_id"])
        self.assertEqual(events[0]["sequence"], 4)
        self.assertEqual(events[-1]["type"], "finished")
        adapter.result_override = {"text": "valid", "segments": [{"start": float("nan")}]}
        with self.assertRaises(ValueError):
            engine.transcribe(wav_frame())

    def test_text_complete_chat_and_stateless_inputs(self):
        engine, adapter, _, _ = self.engine("text_graph")
        self.assertEqual(engine.modality, "text")
        reply = engine.complete(b"hello", max_new_tokens=16)
        self.assertEqual(reply["text"], "reply:hello")
        self.assertEqual(reply["max_new_tokens"], 16)
        self.assertFalse(reply["stateful_chat"])
        self.assertFalse(reply["execution_devices_verified"])
        self.assertIsNone(reply["executed_device"])
        self.assertEqual(reply["compiled_device"], "fixture-CPU")
        self.assertEqual(engine.chat([{"role": "user", "content": "one"}])["text"], "reply:one")
        self.assertEqual(engine.chat([{"role": "user", "content": "two"}])["text"], "reply:two")
        self.assertEqual(adapter.calls[1][0], [{"role": "user", "content": "one"}])
        self.assertEqual(adapter.calls[2][0], [{"role": "user", "content": "two"}])
        with self.assertRaises(ValueError):
            engine.create_stream("text")
        with self.assertRaises(ValueError):
            engine.create_stream("audio")

    def test_text_invalid_payload_messages_tokens_and_recovery(self):
        engine, adapter, _, _ = self.engine("text_graph")
        for prompt in (b"", "  ", b"\xff", "\ud800", bytearray(b"hello"), "x" * 4097, "\u00e9" * 2049):
            with self.assertRaises(ValueError):
                engine.complete(prompt)
        for tokens in (True, 0, -1, 257, float("nan"), "16"):
            with self.assertRaises(ValueError):
                engine.complete("hello", max_new_tokens=tokens)
        for messages in ([], {}, [{"role": "tool", "content": "x"}],
                         [{"role": "user", "content": ["x"]}],
                         [{"role": "user", "content": ""}],
                         [{"role": "user", "content": "x", "tool_calls": []}],
                         [{"role": "user", "content": "x"}] * 65,
                         [{"role": "user", "content": "x" * 2100}] * 2):
            with self.assertRaises(ValueError):
                engine.chat(messages)
        self.assertEqual(adapter.calls, [])
        adapter.result_override = {"wrong": "no text"}
        with self.assertRaises(ValueError):
            engine.complete("hello")
        adapter.result_override = {"text": "x", "execution_devices_verified": "yes"}
        with self.assertRaises(ValueError):
            engine.complete("hello")
        adapter.result_override = None
        self.assertEqual(engine.complete("recover")["text"], "reply:recover")

    def test_text_cancellation_and_caller_message_aliases(self):
        engine, adapter, _, _ = self.engine("text_graph", block_first=True)
        messages = [{"role": "user", "content": "original"}]
        cancelled, errors = threading.Event(), []
        def request():
            try:
                engine.chat(messages, cancel=cancelled.is_set)
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=request)
        thread.start()
        adapter.entered.get(timeout=1)
        messages[0]["content"] = "mutated"
        messages.append({"role": "assistant", "content": "extra"})
        self.assertEqual(adapter.calls[0][0], [{"role": "user", "content": "original"}])
        cancelled.set()
        adapter.release.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GraphCancelled)
        self.assertIn(True, adapter.cancel_observations)
        self.assertEqual(engine.complete("fresh")["text"], "reply:fresh")

    def test_event_cancellation_is_normalized_and_invalid_values_refused(self):
        engine, adapter, _, _ = self.engine("text_graph")
        cancel = threading.Event()
        self.assertEqual(engine.complete("hello", cancel=cancel)["text"], "reply:hello")
        cancel.set()
        with self.assertRaises(GraphCancelled):
            engine.complete("cancelled", cancel=cancel)
        with self.assertRaises(ValueError):
            engine.complete("bad cancellation", cancel=True)
        self.assertEqual(len(adapter.calls), 1)

    def test_compiled_device_is_not_reported_as_verified_execution(self):
        root, _ = self.model("text_graph")
        adapter = FakeAdapter()
        adapter.info = {"backend": "injected-test-fixture", "device": "GPU.0",
                        "execution_devices_verified": False}
        engine = GraphEngine(root, device="GPU", adapter_factory=lambda *unused: adapter)
        self.addCleanup(engine.close)
        self.assertEqual(engine.info()["requested_device"], "GPU")
        self.assertEqual(engine.info()["compiled_device"], "GPU.0")
        self.assertIsNone(engine.info()["executed_device"])
        self.assertIsNone(engine.complete("hello")["executed_device"])


class ManifestTest(RuntimeFixture):
    def refusal(self, root):
        called = []
        with self.assertRaises((ManifestError, MemoryError, ValueError)):
            GraphEngine(root, adapter_factory=lambda *args: called.append(args))
        self.assertEqual(called, [], "backend factory ran before model validation")

    def test_hash_empty_file_and_metadata_corruption(self):
        root, manifest = self.model()
        asset = root / "weights.bin"
        original = asset.read_bytes()
        asset.write_bytes(b"changed fixture")
        self.refusal(root)
        asset.write_bytes(b"")
        self.refusal(root)
        asset.write_bytes(original)
        for field, value in (("schema", "wrong/v1"), ("family", "speech2"), ("adapter", "fake"),
                             ("model_id", "non ASCII \u2603"), ("model_id", ""), ("files", {}),
                             ("max_input_bytes", True), ("resident_bytes", -1),
                             ("workspace_bytes", 0.5), ("options", [])):
            changed = dict(manifest); changed[field] = value
            self.write_manifest(root, changed)
            self.refusal(root)

    def test_traversal_case_aliases_and_reserved_names(self):
        root, manifest = self.model()
        digest = manifest["files"]["weights.bin"]
        for path in ("../weights.bin", "/weights.bin", "C:/weights.bin", "sub\\weights.bin",
                     "sub/../weights.bin", "./weights.bin", "sub//weights.bin", "NUL.bin", "weights.bin "):
            changed = dict(manifest); changed["files"] = {path: digest}
            self.write_manifest(root, changed)
            self.refusal(root)
        changed = dict(manifest); changed["files"] = {"weights.bin": digest, "WEIGHTS.BIN": digest}
        self.write_manifest(root, changed)
        self.refusal(root)
        changed["files"] = {"colibri_runtime.json": digest}
        self.write_manifest(root, changed)
        self.refusal(root)

    def test_duplicate_nonfinite_and_oversized_json(self):
        root, manifest = self.model()
        path = root / "colibri_runtime.json"
        path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
        self.refusal(root)

        for value in (float("nan"), float("inf")):
            changed = dict(manifest); changed["provenance"] = {"value": value}
            self.write_manifest(root, changed)
            self.refusal(root)
        raw = json.dumps(manifest).replace('"test fixture"', '1e9999')
        path.write_text(raw, encoding="utf-8")
        self.refusal(root)
        changed = dict(manifest); changed["provenance"] = {"source": "x" * 65536}
        self.write_manifest(root, changed)
        self.refusal(root)

    def test_text_option_bounds_are_checked_before_factory(self):
        root, manifest = self.model("text_graph")
        for name, value in (("max_tokens", True), ("max_tokens", 257),
                            ("max_prompt_tokens", 4097), ("max_context_tokens", 1)):
            changed = dict(manifest); changed["options"] = dict(manifest["options"], **{name: value})
            self.write_manifest(root, changed)
            self.refusal(root)

    def test_metadata_lookup_has_no_asset_hash_or_memory_admission(self):
        root, manifest = self.model("text_graph")
        manifest["resident_bytes"] = 2 << 30
        self.write_manifest(root, manifest)
        # A metadata lookup neither hashes nor opens any model asset. It still
        # reads/validates the bounded descriptor itself.
        with patch("stream_runtime.hashlib.sha256", side_effect=AssertionError("unexpected asset hash")):
            self.assertEqual(read_manifest_metadata(root), manifest)
        with self.assertRaises(MemoryError):
            validate_manifest(root)
        (root / "weights.bin").unlink()
        self.assertEqual(read_manifest_metadata(root)["model_id"], manifest["model_id"])
        with self.assertRaises(ManifestError):
            validate_manifest(root, memory_budget_bytes=4 << 30)

    def test_metadata_rejects_invalid_unicode_and_descriptor_aliases(self):
        root, manifest = self.model()
        changed = dict(manifest); changed["provenance"] = {"source": "\ud800"}
        self.write_manifest(root, changed)
        with self.assertRaises(ManifestError):
            read_manifest_metadata(root)
        changed = dict(manifest); changed["model_id"] = "contains spaces"
        self.write_manifest(root, changed)
        with self.assertRaises(ManifestError):
            read_manifest_metadata(root)

    def test_symlink_assets_and_parents(self):
        root, manifest = self.model()
        try:
            (root / "link.bin").symlink_to(root / "weights.bin")
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        digest = manifest["files"]["weights.bin"]
        changed = dict(manifest); changed["files"] = {"link.bin": digest}
        self.write_manifest(root, changed)
        self.refusal(root)
        (root / "sub").mkdir()
        (root / "sub" / "asset.bin").write_bytes((root / "weights.bin").read_bytes())
        (root / "linked").symlink_to(root / "sub", target_is_directory=True)
        changed["files"] = {"linked/asset.bin": digest}
        self.write_manifest(root, changed)
        self.refusal(root)

    def test_hardlink_aliases(self):
        root, manifest = self.model()
        try:
            os.link(root / "weights.bin", root / "alias.bin")
        except OSError as error:
            self.skipTest(f"hardlink creation unavailable: {error}")
        manifest["files"]["alias.bin"] = manifest["files"]["weights.bin"]
        self.write_manifest(root, manifest)
        self.refusal(root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "host has no FIFO creation")
    def test_fifo_refused_without_opening_or_blocking(self):
        root, manifest = self.model()
        os.mkfifo(root / "pipe.bin")
        manifest["files"] = {"pipe.bin": "0" * 64}
        self.write_manifest(root, manifest)
        started = time.monotonic()
        self.refusal(root)
        self.assertLess(time.monotonic() - started, 0.2)


if __name__ == "__main__":
    unittest.main()

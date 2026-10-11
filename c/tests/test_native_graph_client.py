"""Bounded native-process transport gates; no model, SDK or image math."""
import base64
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_graph as native
from stream_models import StreamCancelled
from test_native_decode_bench import Pipe, Sink


def manifest(camera=False):
    return {"family": "camera_upscale" if camera else "text_graph", "model_id": "fixture-model",
            "adapter": "super_resolution_openvino" if camera else "text_openvino", "precision": "fixture-f32",
            "resident_bytes": 64, "workspace_bytes": 64, "max_input_bytes": 4096,
            "options": {"max_output_bytes": 4096, "max_output_pixels": 64}, "_native_memory_budget": 8192}


def ready(camera=False, **changes):
    result = {"inference_runtime": "native-c", "backend": "openvino-native-c", "device": "CPU",
              "compiled_device": "CPU", "execution_devices_verified": False, "model_id": "fixture-model",
              "family": "camera_upscale" if camera else "text_graph", "stateful_chat": False}
    result.update(changes)
    return b"READY " + json.dumps(result, separators=(",", ":")).encode() + b"\n"


def reply(request_id="native1", result=None, body=b"", raw=None):
    result = {"text": "English 🦜", "device": "CPU", "execution_devices_verified": False,
              "inference_runtime": "native-c", "segments": []} if result is None else result
    metadata = raw if raw is not None else json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
    return f"RESULT {request_id} {len(metadata)} {len(body)}\n".encode() + metadata + b"\n" + body + b"\n"


class NotifySink(Sink):
    def __init__(self, process):
        super().__init__(); self.process = process
    def write(self, data):
        if self.process.quit_failure and data == b"QUIT\n":
            raise BrokenPipeError("fixture QUIT pipe failure")
        result = super().write(data)
        if self.process.on_write is not None:
            self.process.on_write(data)
        return result


class GraphProcess:
    def __init__(self, initial=None, *, hang=False, stderr=b"", kill_failures=0, quit_failure=False):
        self.stdout = Pipe(ready() if initial is None else initial, short=1)
        self.stderr = io.BytesIO(stderr); self.stdin = NotifySink(self)
        self.alive, self.hang, self.killed = True, hang, False
        self.kill_failures, self.quit_failure = kill_failures, quit_failure
        self.kill_calls, self.wait_calls = 0, 0
        self.on_write = None
    def poll(self): return None if self.alive else (-9 if self.killed else 0)
    def kill(self):
        self.kill_calls += 1
        if self.kill_failures:
            self.kill_failures -= 1
            raise OSError("fixture kill refusal")
        self.killed = True; self.alive = False; self.stdout.finish()
    def wait(self, timeout):
        self.wait_calls += 1
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired("fixture", timeout)
        self.alive = False; self.stdout.finish(); return self.poll()
    def settle_fixture(self):
        self.kill_failures = 0
        if self.alive: self.kill()
        self.stdout.close(); self.stderr.close(); self.stdin.close()


class ValidationProcess(GraphProcess):
    def __init__(self, output, code=0, **kwargs):
        super().__init__(output, **kwargs)
        self.code = code
        # Large malformed-output fixtures exercise the byte limit, rather than
        # spending the request budget on tens of thousands of one-byte reads.
        # Framed transport short reads are covered independently above.
        self.stdout.short = 4096
        self.stdout.finish()
    def poll(self): return None if self.alive else (-9 if self.killed else self.code)


class ClientTest(unittest.TestCase):
    def adapter(self, process=None, *, processes=None, camera=False, **kwargs):
        process = process or GraphProcess(ready(camera))
        candidates = processes or [process]
        launch = patch.object(native.subprocess, "Popen", side_effect=candidates).start()
        self.addCleanup(patch.stopall)
        for candidate in candidates: self.addCleanup(candidate.settle_fixture)
        timeouts = {"startup_timeout": 0.05, "request_timeout": 0.05, "close_timeout": 0.01, **kwargs}
        adapter = native.NativeGraphAdapter(manifest(camera), "fixture-root", binary="fixture-native", **timeouts)
        def cleanup():
            try: adapter.close()
            except (OSError, RuntimeError, subprocess.TimeoutExpired): pass
        self.addCleanup(cleanup)
        return adapter, process, launch

    def test_ready_identity_and_request_wire_are_native_and_exact(self):
        adapter, process, launch = self.adapter()
        self.assertEqual(adapter.info["inference_runtime"], "native-c")
        self.assertEqual(adapter.info["model_id"], "fixture-model")
        process.stdout.feed(reply())
        result = adapter.complete("English 🦜", max_new_tokens=7)
        self.assertEqual(result["text"], "English 🦜")
        self.assertEqual(result["backend"], "openvino-native-c")
        wire = process.stdin.getvalue()
        header, metadata, payload, end = wire.split(b"\n")
        self.assertEqual(header.split()[:2], [b"RUN", b"native1"])
        self.assertEqual(int(header.split()[2]), len(metadata))
        self.assertEqual(int(header.split()[3]), len(payload))
        self.assertEqual(json.loads(metadata), {"operation": "complete", "max_new_tokens": 7})
        self.assertEqual(payload, "English 🦜".encode()); self.assertEqual(end, b"")
        self.assertNotIn("shell", launch.call_args.kwargs)
        self.assertEqual(launch.call_args.args[0][-2:], ["--memory-budget", "8192"])

    def test_ready_wrong_identity_or_missing_backend_refuses_owned_process(self):
        for changes in ({"model_id": "another-model"}, {"family": "speech"}, {"backend": ""},
                        {"inference_runtime": "python"}, {"execution_devices_verified": 1}, {"device": "AUTO"}):
            with self.subTest(changes=changes):
                process = GraphProcess(ready(**changes)); self.addCleanup(process.settle_fixture)
                with patch.object(native.subprocess, "Popen", return_value=process):
                    with self.assertRaises(ValueError):
                        native.NativeGraphAdapter(manifest(), "fixture", binary="fixture",
                            startup_timeout=.05, request_timeout=.05, close_timeout=.01)
                self.assertTrue(process.killed); self.assertFalse(process.alive)
                self.assertTrue(process.stdout.closed and process.stdin.closed)

    def test_explicit_device_substitution_refused_at_ready_and_result(self):
        process = GraphProcess(ready(device="CPU")); self.addCleanup(process.settle_fixture)
        with patch.object(native.subprocess, "Popen", return_value=process), self.assertRaisesRegex(ValueError, "device|substitut"):
            native.NativeGraphAdapter(manifest(), "fixture", device="GPU", binary="fixture", close_timeout=.01)
        self.assertTrue(process.killed)
        adapter, process, _ = self.adapter(GraphProcess(ready(device="GPU.0", compiled_device="GPU.0")), device="GPU")
        process.stdout.feed(reply(result={"text": "bad substitution", "device": "CPU", "execution_devices_verified": True,
                                          "inference_runtime": "native-c"}))
        with self.assertRaisesRegex(ValueError, "device|substitut"): adapter.complete("hello")
        self.assertFalse(process.alive)

    def test_fragmented_png_body_and_metadata_keep_exact_bytes(self):
        adapter, process, _ = self.adapter(camera=True)
        body = b"\x89PNG\r\n\x1a\nfixture\0\xff\nbody"
        process.stdout.feed(reply(result={"width": 4, "height": 4, "scale": 4, "device": "CPU",
            "execution_devices_verified": False, "inference_runtime": "native-c"}, body=body))
        result = adapter.upscale(b"bounded input fixture")
        self.assertEqual(base64.b64decode(result["png_base64"]), body)
        self.assertEqual((result["width"], result["height"], result["scale"]), (4, 4, 4))

    def test_result_identity_cannot_change_verification_backend_or_model(self):
        for changes in ({"execution_devices_verified": True}, {"execution_devices_verified": 0},
                        {"backend": "python"}, {"compiled_device": "GPU.0"}, {"model_id": "other"},
                        {"family": "speech"}, {"inference_runtime": "python"}):
            with self.subTest(changes=changes):
                adapter, process, _ = self.adapter()
                result = {"text": "untrusted", "device": "CPU", "execution_devices_verified": False,
                          "inference_runtime": "native-c", **changes}
                process.stdout.feed(reply(result=result))
                with self.assertRaises(ValueError): adapter.complete("hello")
                self.assertFalse(process.alive)

    def test_malformed_duplicate_nonfinite_and_wrong_id_results_kill(self):
        frames = [reply(raw=b'{"text":"x","text":"y"}'), reply(raw=b'{"text":"x","score":NaN}'),
                  reply(raw=b'{"text":"x","score":1e999}'), reply(request_id="native99"),
                  reply(raw=b'["x"]'), reply(raw=b'{"text":"\xff"}'),
                  reply(raw=b'{"text":"\\ud800"}'), reply(body=b"unexpected"),
                  b"RESULT native1 65537 0\n", b"RESULT native1 1 134217729\n",
                  b"RESULT native1 2 0\n{}!\n", b"unrecognized\n"]
        for frame in frames:
            with self.subTest(frame=frame[:80]):
                adapter, process, _ = self.adapter(); process.stdout.feed(frame)
                with self.assertRaises((ValueError, RuntimeError)): adapter.complete("hello")
                self.assertTrue(process.killed); self.assertFalse(process.alive)
                adapter.close()

    def test_truncated_result_body_reports_parse_error(self):
        adapter, process, _ = self.adapter()
        process.stdout.feed(b"RESULT native1 2 4\n{}\nab"); process.stdout.finish()
        with self.assertRaisesRegex(ValueError, "truncated"): adapter.complete("hello")
        self.assertFalse(process.alive)

    def test_startup_and_request_timeout_drain_owned_process(self):
        process = GraphProcess(b"", hang=True); self.addCleanup(process.settle_fixture)
        with patch.object(native.subprocess, "Popen", return_value=process), self.assertRaises(TimeoutError):
            native.NativeGraphAdapter(manifest(), "fixture", binary="fixture", startup_timeout=.01, close_timeout=.01)
        self.assertTrue(process.killed); self.assertFalse(process.alive)
        adapter, process, _ = self.adapter(GraphProcess(hang=True))
        old_threads = tuple(adapter._threads)
        with self.assertRaises(TimeoutError): adapter.complete("hello")
        self.assertFalse(process.alive); self.assertTrue(process.killed)
        self.assertTrue(all(not thread.is_alive() for thread in old_threads))
        self.assertIsNone(adapter._process)

    def test_event_cancellation_and_lazy_restart_cannot_publish_stale_results(self):
        first, second = GraphProcess(), GraphProcess()
        cancel = threading.Event()
        first.on_write = lambda data: cancel.set() if data.startswith(b"RUN ") else None
        second.on_write = lambda data: second.stdout.feed(reply("native2", result={"text": "fresh", "device": "CPU",
            "execution_devices_verified": False, "inference_runtime": "native-c"})) if data.startswith(b"RUN ") else None
        adapter, _, launch = self.adapter(first, processes=[first, second])
        old_threads, old_queue = tuple(adapter._threads), adapter._frames
        with self.assertRaises(StreamCancelled): adapter.complete("first", cancel=cancel)
        self.assertTrue(first.killed); self.assertTrue(all(not thread.is_alive() for thread in old_threads))
        # Buffered/late old output stays associated with the disposed generation.
        first.stdout.feed(reply("native2", result={"text": "stale"}))
        result = adapter.complete("second")
        self.assertEqual(result["text"], "fresh"); self.assertEqual(launch.call_count, 2)
        self.assertIsNot(adapter._frames, old_queue)

    def test_auto_restart_result_reports_new_generation_compiled_device(self):
        first = GraphProcess()
        second = GraphProcess(ready(device="GPU.0", compiled_device="GPU.0", execution_devices_verified=True))
        cancel = threading.Event()
        first.on_write = lambda data: cancel.set() if data.startswith(b"RUN ") else None
        second.on_write = lambda data: second.stdout.feed(reply("native2", result={"text": "new device", "device": "GPU.0",
            "execution_devices_verified": True, "inference_runtime": "native-c"})) if data.startswith(b"RUN ") else None
        adapter, _, _ = self.adapter(first, processes=[first, second])
        with self.assertRaises(StreamCancelled): adapter.complete("first", cancel=cancel)
        result = adapter.complete("second")
        self.assertEqual((result["device"], result["compiled_device"]), ("GPU.0", "GPU.0"))
        self.assertTrue(result["execution_devices_verified"])
        self.assertEqual(result["model_id"], "fixture-model")

    def test_close_is_idempotent_and_graceful_timeout_reports_forced_cleanup(self):
        adapter, process, _ = self.adapter(); threads = tuple(adapter._threads)
        adapter.close(); adapter.close()
        self.assertFalse(process.alive); self.assertIn(b"QUIT\n", process.stdin.written)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        with self.assertRaisesRegex(RuntimeError, "closed"): adapter.complete("again")
        adapter, process, _ = self.adapter(GraphProcess(hang=True))
        with self.assertRaises(RuntimeError): adapter.close()
        self.assertTrue(process.killed); self.assertFalse(process.alive); self.assertIsNone(adapter._process)

    def test_failed_force_kill_retains_live_process_for_close_retry(self):
        adapter, process, launch = self.adapter(GraphProcess(kill_failures=3))
        with self.assertRaises((TimeoutError, OSError)): adapter.complete("will timeout")
        self.assertTrue(process.alive)
        self.assertIs(adapter._process, process, "live child ownership cannot be dropped after kill fails")
        old_wire = process.stdin.getvalue()
        with self.assertRaises(OSError): adapter.complete("must not restart yet")
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(process.stdin.getvalue(), old_wire)
        with self.assertRaises(OSError): adapter.close()
        self.assertIs(adapter._process, process)
        adapter.close(); self.assertFalse(process.alive)

    def test_real_full_stdin_pipe_obeys_timeout_and_cancellation(self):
        # The actual child publishes READY but never reads stdin. A payload
        # larger than the OS pipe buffer reproduces a synchronous write hang.
        real_popen = subprocess.Popen
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                children = []
                script = "import sys,time; sys.stdout.buffer.write(" + repr(ready()) + ");sys.stdout.flush();time.sleep(30)"
                def launch(*args, **kwargs):
                    process = real_popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=kwargs["env"])
                    children.append(process)
                    return process
                config = manifest(); config["max_input_bytes"] = 1 << 20
                with patch.object(native.subprocess, "Popen", side_effect=launch):
                    adapter = native.NativeGraphAdapter(config, "fixture-root", binary="fixture-native",
                        startup_timeout=2, request_timeout=.15, close_timeout=.5)
                    self.addCleanup(adapter.close)
                    event, errors = threading.Event(), []
                    def request():
                        try: adapter.complete("x" * (1 << 20), cancel=event if cancelled else None)
                        except BaseException as error: errors.append(error)
                    caller = threading.Thread(target=request, daemon=True)
                    started = time.monotonic(); caller.start()
                    timer = threading.Timer(.04, event.set) if cancelled else None
                    if timer: timer.start()
                    caller.join(timeout=2)
                    hung = caller.is_alive()
                    if hung:
                        # Keep this regression bounded even against the old
                        # implementation; kill the fixture to release its pipe.
                        children[0].kill(); children[0].wait(timeout=2); caller.join(timeout=2)
                    if timer: timer.cancel(); timer.join()
                    self.assertFalse(hung, "request deadline must include blocked stdin writes")
                    self.assertLess(time.monotonic() - started, 2)
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], StreamCancelled if cancelled else TimeoutError)
                    self.assertIsNotNone(children[0].poll())
                    self.assertIsNone(adapter._process)
                    self.assertTrue(all(not t.is_alive() for t in adapter._threads))
                    self.assertTrue(children[0].stdin.closed and children[0].stdout.closed and children[0].stderr.closed)

    def test_failed_quit_write_still_kills_and_drains_owned_child(self):
        adapter, process, _ = self.adapter(GraphProcess(quit_failure=True))
        try: adapter.close()
        except (OSError, RuntimeError): pass
        self.assertFalse(process.alive, "broken QUIT pipe must not abandon a live SDK process")
        self.assertTrue(process.killed)

    def test_startup_cleanup_failure_exposes_owned_adapter_for_retry(self):
        process = GraphProcess(ready(model_id="wrong"), kill_failures=1)
        self.addCleanup(process.settle_fixture)
        with patch.object(native.subprocess, "Popen", return_value=process):
            with self.assertRaises(ValueError) as raised:
                native.NativeGraphAdapter(manifest(), "fixture-root", binary="fixture-native", close_timeout=.01)
        adapter = raised.exception.native_adapter
        self.assertIs(adapter._process, process)
        self.assertTrue(process.alive)
        adapter.close()
        self.assertFalse(process.alive)
        self.assertIsNone(adapter._process)

    def test_thread_start_failure_cleans_process_even_if_one_reader_never_started(self):
        process = GraphProcess(); self.addCleanup(process.settle_fixture)
        original = threading.Thread.start; calls = 0
        def failing_start(thread):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError("fixture reader-start failure")
            return original(thread)
        with patch.object(native.subprocess, "Popen", return_value=process), patch.object(threading.Thread, "start", failing_start):
            with self.assertRaisesRegex(RuntimeError, "reader-start failure"):
                native.NativeGraphAdapter(manifest(), "fixture", binary="fixture", close_timeout=.01)
        self.assertFalse(process.alive); self.assertTrue(process.killed)

    def test_stderr_and_frame_queues_remain_bounded_on_shutdown(self):
        process = GraphProcess(stderr=(b"z" * 4095 + b"\n") * 270)
        adapter, process, _ = self.adapter(process)
        process.stdout.feed(reply("unused", result={"text": "ignored"}) * 10)
        adapter.close()
        self.assertLessEqual(len(adapter.stderr), 2048)
        self.assertLessEqual(sum(len(s.encode()) for s in adapter.stderr), native.MAX_STDERR)
        self.assertLessEqual(adapter._frames.maxsize, 4)

    def test_native_pixel_validation_process_errors_and_shape_bounds(self):
        adapter, _, _ = self.adapter(camera=True, request_timeout=1)
        ok = ValidationProcess(b'{"width":2,"height":2,"output_pixels":64}\n')
        self.addCleanup(ok.settle_fixture)
        with patch.object(native.subprocess, "Popen", return_value=ok) as validate:
            result = adapter.validate_media(b"fixture PNG bytes", "camera")
        self.assertEqual(result["output_pixels"], 64)
        self.assertEqual(ok.stdin.written, b"fixture PNG bytes")
        self.assertIn("--validate", validate.call_args.args[0])
        self.assertEqual(validate.call_args.kwargs["env"], adapter._env)
        self.assertTrue(ok.stdout.closed and ok.stderr.closed)
        bad = [(b"", 2), (b'{"width":2,"width":3}', 0),
               (b'{"width":0,"height":2,"output_pixels":0}', 0),
               (b'{"width":1,"height":1,"output_pixels":17}', 0),
               (b'{"width":true,"height":1,"output_pixels":16}', 0),
               (json.dumps({"width":1,"height":1,"output_pixels":16,"extra":"x"*native.MAX_JSON}).encode(), 0)]
        for output, code in bad:
            process = ValidationProcess(output, code)
            self.addCleanup(process.settle_fixture)
            with self.subTest(output=output[:60]), patch.object(native.subprocess, "Popen", return_value=process):
                with self.assertRaises(ValueError): adapter.validate_media(b"fixture", "camera")
            self.assertFalse(process.alive)
            self.assertIsNone(adapter._validator)
        hung = GraphProcess(b"", hang=True)
        self.addCleanup(hung.settle_fixture)
        timeout_adapter, _, _ = self.adapter(camera=True)
        with patch.object(native.subprocess, "Popen", return_value=hung):
            with self.assertRaises((TimeoutError, subprocess.TimeoutExpired)): timeout_adapter.validate_media(b"fixture", "camera")
        self.assertTrue(hung.killed)
        with self.assertRaises(ValueError): adapter.validate_media(b"fixture", "audio")

    def test_failed_validator_cleanup_retains_child_and_blocks_further_work(self):
        adapter, main, _ = self.adapter(camera=True)
        validator = GraphProcess(b"", hang=True, kill_failures=2)
        self.addCleanup(validator.settle_fixture)
        with patch.object(native.subprocess, "Popen", return_value=validator) as launch:
            with self.assertRaises(TimeoutError): adapter.validate_media(b"fixture", "camera")
            self.assertIs(adapter._validator.process, validator)
            self.assertTrue(validator.alive)
            old_wire = main.stdin.getvalue()
            with self.assertRaises(OSError): adapter.upscale(b"no new work until cleanup")
            self.assertEqual(main.stdin.getvalue(), old_wire)
            self.assertEqual(launch.call_count, 1)
            adapter.close()
        self.assertFalse(main.alive or validator.alive)
        self.assertIsNone(adapter._validator)

    def test_production_factory_and_native_operations_import_no_python_ml_sdk(self):
        # Fresh interpreter prevents a previously cached optional dependency
        # from hiding an import. It instantiates the production factory/client.
        script = r'''
import builtins, io, json, sys
from pathlib import Path
from unittest.mock import patch
forbidden={'torch','numpy','PIL','openvino','openvino_genai','transformers'}
original=builtins.__import__
def checked(name,*a,**kw):
    if name.split('.')[0] in forbidden: raise AssertionError('Python ML import: '+name)
    return original(name,*a,**kw)
builtins.__import__=checked
import native_graph, stream_models, stream_runtime
meta={'inference_runtime':'native-c','backend':'openvino-native-c','device':'CPU','compiled_device':'CPU','execution_devices_verified':False,'model_id':'fixture-model','family':'text_graph'}
answer=json.dumps({'text':'native transport fixture','device':'CPU','execution_devices_verified':False,'inference_runtime':'native-c'}).encode()
class P:
    def __init__(self):
        self.stdout=io.BytesIO(b'READY '+json.dumps(meta).encode()+b'\n'+b'RESULT native1 '+str(len(answer)).encode()+b' 0\n'+answer+b'\n\n')
        self.stderr=io.BytesIO(); self.stdin=io.BytesIO(); self.alive=True
    def poll(self):return None if self.alive else 0
    def wait(self,timeout):self.alive=False;return 0
    def kill(self):self.alive=False
p=P();m={'family':'text_graph','model_id':'fixture-model','adapter':'text_openvino','precision':'fixture','resident_bytes':1,'workspace_bytes':1,'max_input_bytes':1024,'options':{}}
with patch.object(native_graph,'native_binary',return_value=Path('fixture-native')),patch.object(native_graph.subprocess,'Popen',return_value=p):
    a=stream_models.make_adapter(m,Path('.'),'CPU')
    assert isinstance(a,native_graph.NativeGraphAdapter)
    assert a.complete('hello')['text']=='native transport fixture'
    a.close();assert not p.alive
assert not any(k.split('.')[0] in forbidden for k in sys.modules)
print('native factory/transport imports only stdlib')
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(native.__file__).parent,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn(b"imports only stdlib", result.stdout)


if __name__ == "__main__": unittest.main()

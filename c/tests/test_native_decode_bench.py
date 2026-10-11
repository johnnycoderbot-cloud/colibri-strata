"""Bounded protocol/teardown tests; no model process or performance load."""
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools import native_decode_bench as bench


class Pipe:
    def __init__(self,data=b"",short=3):
        self.buffer=bytearray(data);self.short=short;self.eof=False;self.closed=False
        self.condition=threading.Condition()
    def feed(self,data):
        with self.condition:self.buffer.extend(data);self.condition.notify_all()
    def finish(self):
        with self.condition:self.eof=True;self.condition.notify_all()
    def read(self,count):
        with self.condition:
            while not self.buffer and not self.eof:self.condition.wait()
            n=min(count,self.short,len(self.buffer));data=bytes(self.buffer[:n]);del self.buffer[:n];return data
    def readline(self,limit=-1):
        with self.condition:
            while b"\n" not in self.buffer and not self.eof and (limit<0 or len(self.buffer)<limit):self.condition.wait()
            pos=self.buffer.find(b"\n");n=pos+1 if pos>=0 else len(self.buffer)
            if limit>=0:n=min(n,limit)
            data=bytes(self.buffer[:n]);del self.buffer[:n];return data
    def close(self):self.closed=True;self.finish()


class Sink(io.BytesIO):
    def close(self):
        if not self.closed:self.written=self.getvalue()
        super().close()


class Process:
    def __init__(self,ready=True,hang=False,stderr=b""):
        self.stdout=Pipe(bench.READY+b" 0.01\n" if ready else b"")
        self.stderr=io.BytesIO(stderr);self.stdin=Sink();self.alive=True;self.hang=hang;self.killed=False
    def poll(self):return None if self.alive else (-9 if self.killed else 0)
    def wait(self,timeout):
        if self.hang and not self.killed:raise subprocess.TimeoutExpired("fixture",timeout)
        self.alive=False;self.stdout.finish();return self.poll()
    def kill(self):self.killed=True;self.alive=False;self.stdout.finish()


class ReaderTest(unittest.TestCase):
    def engine(self,process=None,**kwargs):
        process=process or Process()
        with patch.object(bench.subprocess,"Popen",return_value=process) as launch:
            engine=bench.NativeEngine("fixture-engine","fixture-model",close_timeout=0,**kwargs)
        self.addCleanup(engine.close)
        return engine,process,launch

    def test_short_reads_keep_body_and_numeric_evidence_exact(self):
        engine,process,launch=self.engine()
        process.stdout.feed(b"ACCEPT bench1\nDATA bench1 6 -0.125 1 7 -0.125\nabcdef\nDONE bench1 STAT 1 12.5\n")
        result=engine.turn("hello",max_tokens=2,logprobs=True)
        self.assertEqual(result["text"],"abcdef");self.assertEqual(result["token_ids"],[7])
        self.assertEqual(result["numeric_evidence"][0]["logprob"],-0.125)
        self.assertEqual(result["numeric_evidence"][0]["emitted_index"],0)
        self.assertNotIn("position",result["numeric_evidence"][0])
        self.assertEqual(result["numeric_evidence"][0]["top_tokens"],[{"token_id":7,"logprob":-0.125}])
        self.assertTrue(result["token_ids_verified"])
        self.assertEqual(result["tokens"],1)
        self.assertIn(b"SUBMIT bench1 0 5 2 0 1 logprobs=1\nhello\n",process.stdin.getvalue())
        self.assertNotIn("shell",launch.call_args.kwargs)

    def test_frame_errors_reach_direct_frontier_consumers(self):
        for payload in (b"DATA bench1 nan\n",b"DATA bench1 4194305\n",b"DATA bench1 1\nx!",b"x"*(bench.MAX_LINE+1)):
            with self.subTest(payload=payload[:32]):
                engine,process,_=self.engine();process.stdout.feed(payload)
                with self.assertRaises(ValueError):engine.frames.get(timeout=1)
                engine.close();self.assertTrue(process.stdout.closed)

    def test_truncated_body_failure_is_not_hidden_as_eof(self):
        engine,process,_=self.engine();process.stdout.feed(b"DATA bench1 4\nab");process.stdout.finish()
        with self.assertRaisesRegex(ValueError,"truncated"):engine.frames.get(timeout=1)

    def test_invalid_request_options_never_consume_sequence_or_write(self):
        engine,process,_=self.engine()
        cases=[{"max_tokens":True},{"max_tokens":0},{"temperature":float("nan")},
               {"temperature":float("inf")},{"top_p":0},{"timeout":float("nan")},
               {"timeout":0},{"cancel_after":True},{"cancel_after":65},
               {"extra":"pin=1\nCANCEL x"},{"extra":"pin=1 pin=0"},
               {"extra":"pin=nan"},{"extra":"pin=2"},{"extra":"unframed"}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):engine.turn("hello",**kwargs)
        self.assertEqual(engine.count,0);self.assertEqual(process.stdin.getvalue(),b"")
        with self.assertRaises(UnicodeError):engine.turn("\ud800")

    def test_nonfinite_or_inconsistent_native_evidence_refuses_and_closes(self):
        for frames in (b"DATA bench1 1 nan 1 7 -0.1\nx\nDONE bench1 STAT 1 1\n",
                       b"DATA bench1 1 -0.1 1 7 nan\nx\nDONE bench1 STAT 1 1\n",
                       b"DATA bench1 1 -0.1 2 7 -0.1\nx\nDONE bench1 STAT 1 1\n",
                       b"DATA bench1 1 -0.1 1 7 -0.1\nx\nDONE bench1 STAT 2 1\n",
                       b"DONE bench1 STAT 1 inf\n",b"DONE bench1\n"):
            with self.subTest(frames=frames):
                engine,process,_=self.engine();process.stdout.feed(frames)
                with self.assertRaises(ValueError):engine.turn("hello",logprobs=True)
                self.assertFalse(process.alive);self.assertTrue(process.stdin.closed)

    def test_startup_and_turn_timeouts_release_owned_process(self):
        process=Process(ready=False,hang=True)
        with patch.object(bench.subprocess,"Popen",return_value=process):
            with self.assertRaises(TimeoutError):
                bench.NativeEngine("fixture","model",startup_timeout=0.01,close_timeout=0)
        self.assertTrue(process.killed);self.assertTrue(process.stdout.closed)
        engine,process,_=self.engine(Process(hang=True))
        with self.assertRaises(TimeoutError):engine.turn("hello",timeout=0.01)
        self.assertTrue(process.killed);self.assertIn(b"CANCEL bench1\n",process.stdin.written)
        self.assertTrue(all(not thread.is_alive() for thread in engine.readers))

    def test_cancelled_partial_result_has_verified_token_evidence(self):
        engine,process,_=self.engine()
        process.stdout.feed(b"DATA bench1 1 -0.1 1 7 -0.1\nx\nERROR bench1 CANCELLED\n")
        result=engine.turn("hello",logprobs=True,cancel_after=1)
        self.assertTrue(result["cancelled"]);self.assertEqual(result["token_ids"],[7])
        self.assertEqual(result["tokens"],1);self.assertTrue(result["token_count_verified"])
        self.assertIn(b"CANCEL bench1\n",process.stdin.getvalue())

    def test_queue_and_stderr_bounds_and_full_queue_close(self):
        engine,process,_=self.engine(Process(stderr=(b"x"*1023+b"\n")*1300))
        process.stdout.feed(b"notice\n"*100)
        deadline=time.monotonic()+1
        while engine.frames.qsize()<16 and time.monotonic()<deadline:time.sleep(0.005)
        self.assertEqual(engine.frames.qsize(),16)
        engine.close();engine.close()
        self.assertTrue(all(not thread.is_alive() for thread in engine.readers))
        self.assertLessEqual(sum(len(s.encode()) for s in engine.stderr),bench.MAX_STDERR)
        self.assertLessEqual(len(engine.stderr),2048);self.assertGreater(engine.stderr_truncated_bytes,0)

    def test_invalid_constructor_bounds_and_env_fail_before_launch(self):
        with patch.object(bench.subprocess,"Popen") as launch:
            for kwargs in ({"threads":True},{"threads":65},{"startup_timeout":float("nan")},
                           {"env":{"bad=name":"value"}},{"env":{"OK":"\0"}},
                           {"cache_per_layer":0},{"cache_per_layer":True},{"expert_bits":17}):
                with self.assertRaises(ValueError):bench.NativeEngine("fixture","model",**kwargs)
            launch.assert_not_called()

    def test_explicit_expert_cache_and_precision_reach_launch_arguments(self):
        _, _, launch = self.engine(cache_per_layer=16, expert_bits=4)
        self.assertEqual(launch.call_args.args[0][-2:], ["16", "4"])


class ReportEngine:
    instances=[];mismatch=False;fail_feature=False
    def __init__(self,engine,*args):
        self.label=Path(engine).name;self.stderr=["fixture tail\n"];self.closed=False
        self.ready_line="fixture READY";self.launch_settings=dict(args[2]);self.env=dict(args[2])
        if self.fail_feature and self.label=="feature":raise RuntimeError("fixture startup failure")
        self.instances.append(self)
    def turn(self,prompt,tokens,logprobs=False):
        different=self.mismatch and self.label=="feature"
        return {"token_ids":[8 if different else 7] if logprobs else [],
                "numeric_frames":["DATA bench1 1 -0.125 1 7 -0.125"] if logprobs else [],
                "numeric_evidence":[{"token_id":7,"logprob":-0.125}] if logprobs else [],
                "output_sha256":"different" if different else "same","tokens":1,
                "engine_decode_tokens_per_second":20 if self.label=="feature" else 10}
    def close(self):self.closed=True


class ReportTest(unittest.TestCase):
    def run_report(self,mismatch=False,fail=False):
        home=tempfile.TemporaryDirectory();self.addCleanup(home.cleanup);root=Path(home.name)
        for name in ("baseline","feature"):(root/name).write_bytes(name.encode())
        model=root/"model";model.mkdir();(model/"config.json").write_text('{"model_type":"fixture"}')
        prompts=root/"prompts.json";prompts.write_text('["hello"]')
        output=root/"report.json";ReportEngine.instances=[];ReportEngine.mismatch=mismatch;ReportEngine.fail_feature=fail
        argv=["--baseline",str(root/"baseline"),"--feature",str(root/"feature"),"--model",str(model),
              "--prompts",str(prompts),"--output",str(output),"--repeats","5"]
        with patch.object(bench,"NativeEngine",ReportEngine),contextlib.redirect_stdout(io.StringIO()):
            if mismatch or fail:
                with self.assertRaises(RuntimeError):bench.main(argv)
            else:bench.main(argv)
        return json.loads(output.read_text()),ReportEngine.instances
    def test_complete_report_retains_actual_numeric_checks_and_identity_scope(self):
        report,engines=self.run_report()
        self.assertEqual(report["status"],"complete");self.assertEqual(len(report["records"]),5)
        self.assertEqual(report["median_paired_decode_ratio"],2)
        self.assertEqual(report["correctness"][0]["baseline"]["numeric_evidence"][0]["logprob"],-0.125)
        self.assertFalse(report["baseline_identity"]["weights_hashed"])
        self.assertFalse(report["feature_identity"]["accelerator_execution_verified"])
        self.assertTrue(all(engine.closed for engine in engines))
    def test_mismatch_retains_both_sides_and_partial_failure_artifact(self):
        report,engines=self.run_report(mismatch=True)
        self.assertEqual(report["status"],"failed");self.assertFalse(report["correctness"][0]["equal"])
        self.assertEqual(report["correctness"][0]["feature"]["token_ids"],[8])
        self.assertTrue(all(engine.closed for engine in engines))
    def test_feature_startup_failure_still_closes_baseline_and_writes_identity(self):
        report,engines=self.run_report(fail=True)
        self.assertEqual(report["status"],"failed");self.assertEqual(report["error"]["type"],"RuntimeError")
        self.assertIn("binary_sha256",report["feature_identity"])
        self.assertTrue(engines[0].closed)
    def test_ambient_feature_and_accelerator_knobs_do_not_contaminate_arms(self):
        inherited={"COLI_CUDA":"1","COLI_VULKAN":"1","COLI_STRATA_HEADS":"stale-bank",
                   "COLI_STRATA_TRACE":"stale-trace","COLI_STRATA_INPUT":"next-embedding",
                   "COLI_SPEC_GATE":"0","QWEN_HEAD_BATCH":"0","QWEN_EXPERT_BATCH":"0"}
        with patch.dict(os.environ,inherited):report,engines=self.run_report()
        for engine in engines:
            for name in ("COLI_CUDA","COLI_VULKAN"):self.assertEqual(engine.env[name],"0")
            for name in ("COLI_STRATA_HEADS","COLI_STRATA_TRACE","COLI_STRATA_INPUT"):self.assertEqual(engine.env[name],"")
            for name in ("COLI_SPEC_GATE","QWEN_HEAD_BATCH","QWEN_EXPERT_BATCH"):self.assertEqual(engine.env[name],"1")
        self.assertEqual(report["baseline_launch_settings"],engines[0].env)


if __name__=="__main__":unittest.main()

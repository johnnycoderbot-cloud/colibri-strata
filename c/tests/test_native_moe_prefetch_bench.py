"""Model-free protocol and failed-qualification regressions for MoE timing."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import native_decode_bench as driver
from tools import native_moe_prefetch_bench as bench
from test_native_decode_bench import Process


class ProtocolTest(unittest.TestCase):
    def engine(self):
        process = Process()
        with patch.object(driver.subprocess, "Popen", return_value=process):
            engine = bench.MoeMetricEngine("fixture", "fixture-model", close_timeout=0)
        self.addCleanup(engine.close)
        return engine, process

    def test_profile_after_done_belongs_to_current_turn(self):
        engine, process = self.engine()
        for i in (1, 2):
            process.stdout.feed(f"ACCEPT bench{i}\nDATA bench{i} 1 -0.1 1 7 -0.1\nx\nDONE bench{i} STAT 1 2 75 1.2 5 1\nPROF 0.5 5 1 0.1 0.0 0.2 0.1 0.1 6\n".encode())
            result = engine.turn("hello", logprobs=True)
            self.assertEqual(result["protocol_profile"]["request_seconds"], 0.5)
            self.assertEqual(result["cache_hit_percent"], 75)
            self.assertEqual(result["engine_request_tokens_per_second"], 2)
            self.assertNotIn("engine_decode_tokens_per_second", result)
            self.assertNotIn("decode_seconds", result["protocol_profile"])
            self.assertEqual(result["request_id"], f"bench{i}")

    def test_missing_mismatched_or_early_profile_refuses_and_closes(self):
        for frames in (b"DONE bench1 STAT 1 2 75 1.2 5 1\nNOTICE\n",
                       b"DONE bench1 STAT 1 2 75 1.2 5 1\nPROF 0.5 6 1 0 0 0 0 0 1\n",
                       b"PROF 0.5 5 1 0 0 0 0 0 1\n"):
            engine, process = self.engine(); process.stdout.feed(frames)
            with self.assertRaises(ValueError): engine.turn("hello")
            self.assertFalse(process.alive)

    def test_prefetch_evidence_requires_drained_queue_and_real_loads(self):
        good = bench.pilot_statistics(["[olmoe PILOT] turn bench1: queued=5 completed=5 loads=3\n"])
        self.assertEqual(good["bench1"]["loads"], 3)
        for lines in (["[olmoe PILOT] turn x: queued=5 completed=4 loads=3"],
                      ["[olmoe PILOT] turn x: queued=1 completed=1 loads=2"],
                      ["[olmoe PILOT] invalid"],
                      ["[olmoe PILOT] turn x: queued=0 completed=0 loads=0"]*2):
            with self.assertRaises(ValueError): bench.pilot_statistics(lines)


class FakeEngine:
    instances = []; mismatch = False; missing = False
    def __init__(self, engine, model, threads, env, wsl, **kwargs):
        self.pilot = int(env["PILOT"]); self.count = 0; self.closed = False
        self.launch_settings = env; self.startup_ready_seconds = 0.01; self.stderr = []
        self.kwargs = kwargs; self.instances.append(self)
    def turn(self, prompt, tokens, logprobs, **kwargs):
        self.count += 1; rid = "bench"+str(self.count)
        if not self.missing:
            self.stderr.append(f"[olmoe PILOT] turn {rid}: queued={self.pilot} completed={self.pilot} loads={self.pilot}\n")
        return {"request_id":rid, "tokens":1, "output_sha256":"bad" if self.mismatch and self.pilot else "same",
                "token_ids_verified":logprobs, "token_ids":[7] if logprobs else [],
                "numeric_frames":[f"DATA {rid} 1 -0.1 1 7 -0.1"] if logprobs else [],
                "engine_request_tokens_per_second":2 if self.pilot else 1, "wall_seconds":0.5 if self.pilot else 1}
    def close(self): self.closed = True


class ReportTest(unittest.TestCase):
    def run_report(self, mismatch=False, missing=False):
        home = tempfile.TemporaryDirectory(); self.addCleanup(home.cleanup); root = Path(home.name)
        engine = root/"engine"; engine.write_bytes(b"binary")
        model = root/"model"; model.mkdir()
        (model/"config.json").write_text('{"model_type":"olmoe"}')
        (model/"tokenizer.json").write_text('{}')
        (model/"model.safetensors").write_bytes(b"fixture")
        prompts = root/"prompts.json"; prompts.write_text('["one","two","three"]')
        output = root/"out.json"
        FakeEngine.instances = []; FakeEngine.mismatch = mismatch; FakeEngine.missing = missing
        with contextlib.redirect_stdout(io.StringIO()):
            args = ["--engine",str(engine),"--model",str(model),"--prompts",str(prompts),"--output",str(output)]
            if mismatch or missing:
                with self.assertRaises(RuntimeError): bench.main(args, engine_factory=FakeEngine)
            else: bench.main(args, engine_factory=FakeEngine)
        self.assertTrue(all(e.closed for e in FakeEngine.instances))
        return json.loads(output.read_text()), FakeEngine.instances

    def test_same_engine_fixed_budget_paired_outputs_and_execution(self):
        report, engines = self.run_report()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["records"]), 15)
        self.assertEqual(report["summary"]["median_paired_request_speed_ratio"], 2)
        self.assertTrue(report["assets_verified_before_and_after"])
        self.assertEqual(engines[0].kwargs, engines[1].kwargs)
        self.assertEqual({k for k in engines[0].launch_settings if engines[0].launch_settings[k] != engines[1].launch_settings[k]}, {"PILOT"})
        self.assertEqual(report["records"][0]["candidate"]["native_prefetch_statistics"]["loads"], 1)

    def test_mismatch_and_missing_execution_leave_failed_reviewable_reports(self):
        for mismatch, missing in ((True, False), (False, True)):
            report, _ = self.run_report(mismatch, missing)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["correctness"])


if __name__ == "__main__": unittest.main()

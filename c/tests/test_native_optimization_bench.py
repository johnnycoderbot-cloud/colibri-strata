"""Benchmark qualification and artifact ownership; no native model execution."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import native_optimization_bench as bench


class FakeEngine:
    def __init__(self, owner, engine, model, threads, env, wsl, *, startup_timeout):
        self.owner = owner
        self.engine = Path(engine)
        self.env = dict(env)
        self.launch_settings = {**self.env, "OMP_NUM_THREADS": str(threads), "SERVE": "1"}
        self.startup_ready_seconds = 0.2 + len(owner.engines)
        self.stderr = ["fixture only; no native launch\n"]
        self.stderr_truncated_bytes = 3
        self.ready_line = "fixture READY"
        self.closed = False
        self.count = 0
        owner.engines.append(self)

    def turn(self, prompt, max_tokens=64, logprobs=False, *, timeout=120):
        if self.closed:
            raise AssertionError("request on closed engine")
        self.count += 1
        self.owner.calls.append((self, prompt, max_tokens, logprobs, timeout))
        text = "x" * 64
        rid = "bench" + str(self.count)
        rate = 40 if self.engine == self.owner.baseline else 45
        rate += 3 * int(self.env["COLI_IDOT_ROWS"])
        rate += 2 * int(self.env["QWEN_PROJECTION_GROUP"])
        rate += 4 * int(self.env["COLI_IDOT_WIDE"])
        rate += bool(self.env["COLI_STRATA_HEADS"])
        rate += 7 * bool(self.env["Q36_MTP"])
        result = {"request_id": rid, "tokens": 64, "engine_decode_tokens_per_second": rate,
                  "wall_seconds": 64 / rate + 0.1, "ttft_seconds": 0.1, "accept_seconds": 0.01,
                  "text": text, "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "token_ids": [7] * 64 if logprobs else [], "token_ids_verified": logprobs,
                  "numeric_frames": [f"DATA {rid} 1 -0.125 1 7 -0.125"] * 64 if logprobs else [],
                  "numeric_evidence": []}
        if self.owner.fault:
            self.owner.fault(self, prompt, logprobs, result)
        if self.env["Q36_MTP"] and not self.owner.missing_mtp_evidence:
            self.stderr.append(f"[qwen36 MTP] turn {rid}: bank_calls=1 bank_seconds=0.01 failures=0 forwards=32 tokens=63 accepted=8 drafts=16 verifies=8 ended_mid_verify=0\n")
        return result

    def close(self):
        self.closed = True
        if self.owner.close_fault:
            self.owner.close_fault(self)


class Factory:
    def __init__(self, baseline):
        self.baseline = baseline
        self.engines = []
        self.calls = []
        self.fault = None
        self.close_fault = None
        self.creation_fault = None
        self.missing_mtp_evidence = False

    def __call__(self, *args, **kwargs):
        if self.creation_fault:
            self.creation_fault(*args, **kwargs)
        return FakeEngine(self, *args, **kwargs)


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.model / "qwen36_meta.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"tiny identity fixture")
        self.baseline = self.root / "baseline"
        self.optimized = self.root / "optimized"
        self.heads = self.root / "heads.snh"
        self.baseline.write_bytes(b"stock identity fixture")
        self.optimized.write_bytes(b"fork identity fixture")
        self.heads.write_bytes(b"head identity fixture")
        self.mtp = self.root / "mtp-sidecar"
        self.mtp.mkdir()
        (self.mtp / "qwen36_mtp.json").write_text("{}", encoding="utf-8")
        (self.mtp / "model-mtp.safetensors").write_bytes(b"MTP identity fixture")
        self.prompts = self.root / "prompts.json"
        self.prompt_values = ["held-out one", "held-out two", "held-out three"]
        self.prompts.write_text(json.dumps(self.prompt_values), encoding="utf-8")
        self.output = self.root / "report.json"
        self.factory = Factory(self.baseline)

    def argv(self, *extra):
        return ["--baseline", str(self.baseline), "--optimized", str(self.optimized),
                "--model", str(self.model), "--prompts", str(self.prompts),
                "--output", str(self.output), *extra]

    def run_bench(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.main(self.argv(*extra), engine_factory=self.factory)

    def failed_report(self):
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(all(engine.closed for engine in self.factory.engines))
        return result

    def test_default_study_is_heads_off_with_alternating_fixed_cap_pairs(self):
        report = self.run_bench()
        study, = report["studies"]
        self.assertEqual(study["status"], "complete")
        self.assertEqual(len(study["correctness"]), 3)
        self.assertTrue(all(row["equal"] for row in study["correctness"]))
        self.assertEqual(len(study["fresh_process_requests"]), 1)
        self.assertFalse(study["fresh_process_requests"][0]["page_cache_cold_verified"])
        self.assertEqual(len(study["warmups"]), 3)
        self.assertEqual(len(study["records"]), 15)
        self.assertEqual(study["summary"]["median_paired_decode_ratio"], 1.25)
        self.assertEqual([item["pairs"] for item in study["summary"]["per_prompt"]], [5, 5, 5])
        for index in range(3):
            rows = [row for row in study["records"] if row["prompt"] == index]
            self.assertEqual([row["order"] for row in rows],
                             [["control", "candidate"] if (index + repeat) % 2 == 0 else
                              ["candidate", "control"] for repeat in range(5)])
        self.assertTrue(all(call[2] == 64 for call in self.factory.calls))
        self.assertTrue(all(not row[side]["numeric_frames"] for row in study["records"]
                            for side in ("control", "candidate")))
        self.assertTrue(all(engine.closed for engine in self.factory.engines))
        self.assertTrue(report["model_identity_verified_before_and_after"])
        self.assertFalse(report["hardware_execution_qualified"])
        self.assertFalse(report["prefill_metric_available"])
        self.assertEqual(report["settings"]["wide"], 0)
        self.assertTrue(all(engine.env["COLI_IDOT_WIDE"] == "0" for engine in self.factory.engines))
        self.assertEqual(report, json.loads(self.output.read_text(encoding="utf-8")))

    def test_ablations_change_only_one_new_knob_and_heads_use_same_binary(self):
        report = self.run_bench("--ablate-knobs", "--heads", str(self.heads))
        self.assertEqual(len(report["studies"]), 4)
        for study, key in zip(report["studies"][1:3], ("COLI_IDOT_ROWS", "QWEN_PROJECTION_GROUP")):
            self.assertEqual(study["control"]["engine"], str(self.optimized))
            self.assertEqual(study["candidate"]["engine"], str(self.optimized))
            self.assertEqual({k for k in study["control"]["env"]
                              if study["control"]["env"][k] != study["candidate"]["env"][k]}, {key})
        heads = report["studies"][3]
        self.assertEqual(heads["control"]["engine"], heads["candidate"]["engine"])
        self.assertEqual({k for k in heads["control"]["env"]
                          if heads["control"]["env"][k] != heads["candidate"]["env"][k]},
                         {"COLI_STRATA_HEADS", "COLI_STRATA_INPUT"})
        self.assertEqual(report["studies"][0]["candidate"]["startup_ready_seconds"], 1.2)
        self.assertEqual(heads["control"]["startup_ready_seconds"], 6.2)
        for study in report["studies"]:
            for side in ("control", "candidate"):
                env = study[side]["env"]
                self.assertEqual(env["QWEN_HEAD_BATCH"], "1")
                self.assertEqual(env["QWEN_EXPERT_BATCH"], "1")
                self.assertEqual(env["COLI_LOOKUP"], "1")
        self.assertEqual(report["head_identity"]["sha256"], bench.file_sha(self.heads))

    def test_ambient_acceleration_and_force_settings_are_overridden(self):
        with patch.dict(os.environ, {"COLI_LOOKUP_FORCE": "bogus", "COLI_CUDA": "1",
                                     "COLI_STRATA_HEADS": "outside-bank", "COLI_LOOKUP_DRAFTS": "1", "COLI_IDOT_WIDE": "1"}):
            report = self.run_bench("--rows", "0", "--projection-group", "0", "--phase-timers")
        for side in ("control", "candidate"):
            env = report["studies"][0][side]["env"]
            self.assertEqual(env["COLI_CUDA"], "0")
            self.assertEqual(env["COLI_LOOKUP_FORCE"], "")
            self.assertEqual(env["COLI_LOOKUP_DRAFTS"], "")
            self.assertEqual(env["COLI_STRATA_HEADS"], "")
            self.assertEqual(env["COLI_TIMERS"], "1")
            self.assertEqual(env["COLI_IDOT_WIDE"], "0")

    def test_combined_mtp_uses_unchanged_stock_control_and_actual_candidate(self):
        report=self.run_bench("--mtp-dir",str(self.mtp),"--studies","stock-mtp")
        study=report["studies"][0]
        self.assertEqual(study["control"]["engine"],str(self.baseline))
        self.assertEqual(study["candidate"]["engine"],str(self.optimized))
        self.assertFalse(study["control"]["pretrained_mtp"])
        self.assertTrue(study["candidate"]["pretrained_mtp"])
        self.assertTrue(study["pretrained_mtp_executed"])
        self.assertEqual(study["control"]["env"]["Q36_MTP"],"")
        self.assertTrue(study["candidate"]["env"]["Q36_MTP"])

    def test_wide_ablation_is_independent_and_other_ablations_hold_wide_fixed(self):
        report = self.run_bench("--wide", "1", "--ablate-knobs", "--ablate-wide", "--heads", str(self.heads))
        self.assertEqual(report["settings"]["studies"], ["stock", "rows", "group", "wide", "heads"])
        self.assertEqual(report["settings"]["wide"], 1)
        self.assertEqual(report["studies"][0]["control"]["env"]["COLI_IDOT_WIDE"], "0")
        self.assertEqual(report["studies"][0]["candidate"]["env"]["COLI_IDOT_WIDE"], "1")
        wide = report["studies"][3]
        self.assertEqual(wide["control"]["engine"], str(self.optimized))
        self.assertEqual(wide["candidate"]["engine"], str(self.optimized))
        self.assertEqual({key for key in wide["control"]["env"]
                          if wide["control"]["env"][key] != wide["candidate"]["env"][key]}, {"COLI_IDOT_WIDE"})
        for study in (report["studies"][1], report["studies"][2], report["studies"][4]):
            for side in ("control", "candidate"):
                self.assertEqual(study[side]["wide"], 1)
                self.assertEqual(study[side]["env"]["COLI_IDOT_WIDE"], "1")

    def test_study_selection_runs_only_requested_studies_in_requested_order(self):
        report = self.run_bench("--wide", "1", "--ablate-knobs", "--heads", str(self.heads), "--studies", "wide,stock")
        self.assertEqual(report["settings"]["studies"], ["wide", "stock"])
        self.assertEqual([study["id"] for study in report["studies"]], ["wide", "stock"])
        self.assertEqual(len(self.factory.engines), 4)
        self.assertTrue(all(engine.env["COLI_STRATA_HEADS"] == "" for engine in self.factory.engines))

    def test_heads_only_selection_uses_chosen_wide_in_both_arms(self):
        report = self.run_bench("--wide", "1", "--heads", str(self.heads), "--studies", "heads")
        study, = report["studies"]
        self.assertEqual(study["id"], "heads")
        self.assertEqual(study["control"]["engine"], study["candidate"]["engine"])
        self.assertEqual(study["control"]["env"]["COLI_IDOT_WIDE"], "1")
        self.assertEqual(study["candidate"]["env"]["COLI_IDOT_WIDE"], "1")

    def test_invalid_study_selection_never_claims_output_or_launches(self):
        for selection in ("", "stock,stock", "stock,unknown", "stock, wide", "heads", "mtp"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.run_bench("--studies", selection)
            self.assertFalse(self.output.exists())
        self.assertEqual(self.factory.engines, [])

    def test_mtp_study_is_same_binary_with_depth_fixed_and_ambient_force_cleared(self):
        with patch.dict(os.environ, {"Q36_MTP": "ambient-sidecar", "Q36_MTP_FORCE": "accept", "Q36_MTP_TEST_ORACLE": "oracle.json"}):
            report = self.run_bench("--mtp-dir", str(self.mtp), "--mtp-drafts", "3", "--studies", "stock,mtp")
        study = report["studies"][1]
        self.assertTrue(study["pretrained_mtp_executed"])
        self.assertEqual(study["control"]["engine"], study["candidate"]["engine"])
        self.assertEqual({key for key in study["control"]["env"] if study["control"]["env"][key] != study["candidate"]["env"][key]}, {"Q36_MTP"})
        for definition in report["studies"]:
            for side in ("control", "candidate"):
                env = definition[side]["env"]
                self.assertEqual(env["Q36_MTP_DRAFTS"], "3")
                self.assertEqual(env["Q36_MTP_FORCE"], "")
                self.assertEqual(env["Q36_MTP_TEST_ORACLE"], "")
        self.assertEqual(report["studies"][0]["candidate"]["env"]["Q36_MTP"], "")
        self.assertEqual(len(study["records"]), 15)
        self.assertEqual(study["records"][0]["candidate"]["native_mtp_statistics"]["bank_calls"], 1)
        self.assertTrue(report["mtp_identity"]["manifest_sha256"])

    def test_missing_mtp_execution_evidence_fails_instead_of_claiming_speedup(self):
        self.factory.missing_mtp_evidence = True
        with self.assertRaisesRegex(RuntimeError, "MTP execution evidence"):
            self.run_bench("--mtp-dir", str(self.mtp), "--studies", "mtp")
        report = self.failed_report()
        self.assertEqual(report["studies"][0]["status"], "failed")

    def test_mtp_asset_mutation_and_protected_report_path(self):
        self.output = self.mtp / "report.json"
        with self.assertRaisesRegex(ValueError, "outside model"):
            self.run_bench("--mtp-dir", str(self.mtp), "--studies", "mtp")
        self.assertEqual(self.factory.engines, [])
        self.output = self.root / "report.json"
        self.factory.close_fault = lambda engine: (self.mtp / "model-mtp.safetensors").write_bytes(b"mutated")
        with self.assertRaisesRegex(RuntimeError, "MTP sidecar changed"):
            self.run_bench("--mtp-dir", str(self.mtp), "--studies", "mtp")
        self.failed_report()

    def test_wsl_bank_path_and_source_description_are_recorded(self):
        report = self.run_bench("--wsl", "kali-pc", "--heads", str(self.heads),
                                "--optimized-source-kind", "candidate commit with new CPU kernels")
        side = report["studies"][-1]["candidate"]
        self.assertEqual(side["env"]["COLI_STRATA_HEADS"], bench.linux_path(self.heads))
        self.assertEqual(side["source_kind"], "candidate commit with new CPU kernels")
        self.assertIn("--wsl", report["invocation"]["arguments"])
        self.assertTrue(report["model_identity"]["hashing_precedes_measurement_and_warms_page_cache"])
        names = {asset["name"] for asset in report["model_identity"]["files"]}
        self.assertIn("model.safetensors.index.json", names)

    def test_invalid_parameters_and_prompt_types_never_claim_output_or_launch(self):
        for option in (("--repeats", "4"), ("--tokens", "32"), ("--threads", "0"),
                       ("--request-timeout", "nan"), ("--startup-timeout", "0")):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    self.run_bench(*option)
                self.assertFalse(self.output.exists())
        for prompts in (["a", "b"], ["a", "a", "b"], [[], "a", "b"], ["", "a", "b"],
                        ["a" * (bench.MAX_PROMPT_BYTES + 1), "b", "c"]):
            with self.subTest(prompts=str(prompts)[:40]):
                self.prompts.write_text(json.dumps(prompts), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.run_bench()
                self.assertFalse(self.output.exists())
        self.assertEqual(self.factory.engines, [])

    def test_existing_artifact_and_model_directory_are_preserved(self):
        self.output.write_bytes(b"original artifact")
        with self.assertRaises(FileExistsError):
            self.run_bench()
        self.assertEqual(self.output.read_bytes(), b"original artifact")
        self.output = self.model / "new-report.json"
        with self.assertRaisesRegex(ValueError, "outside model"):
            self.run_bench()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.factory.engines, [])

    def test_token_and_logprob_mismatch_prevents_timed_campaign(self):
        def fault(engine, prompt, logprobs, result):
            if engine.engine != self.baseline and logprobs:
                result["numeric_frames"][0] = f"DATA {result['request_id']} 1 -0.25 1 7 -0.25"
        self.factory.fault = fault
        with self.assertRaisesRegex(RuntimeError, "token ID/logprob/output mismatch"):
            self.run_bench()
        report = self.failed_report()
        study = report["studies"][0]
        self.assertFalse(study["correctness"][0]["equal"])
        self.assertEqual(study["records"], [])
        self.assertEqual(study["warmups"], [])
        a = study["correctness"][0]["control"]
        b = copy.deepcopy(a)
        b["token_ids"][0] = 8
        self.assertFalse(bench.check_numeric(a, b))
        b = copy.deepcopy(a)
        b["token_ids_verified"] = False
        self.assertFalse(bench.check_numeric(a, b))
        b = copy.deepcopy(a)
        b["numeric_frames"].pop()
        self.assertFalse(bench.check_numeric(a, b))

    def test_timed_output_mismatch_retains_partial_record(self):
        def fault(engine, prompt, logprobs, result):
            if engine.engine != self.baseline and engine.count == 4:
                result["output_sha256"] = "different"
        self.factory.fault = fault
        with self.assertRaisesRegex(RuntimeError, "timed output/count"):
            self.run_bench()
        report = self.failed_report()
        self.assertEqual(len(report["studies"][0]["records"]), 1)
        self.assertEqual(len(report["studies"][0]["correctness"]), 1)

    def test_startup_failure_closes_already_started_arm_and_retains_identity(self):
        def failure(engine, *args, **kwargs):
            if Path(engine) == self.optimized:
                raise TimeoutError("fixture startup timed out")
        self.factory.creation_fault = failure
        with self.assertRaisesRegex(TimeoutError, "startup timed out"):
            self.run_bench()
        report = self.failed_report()
        self.assertEqual(len(self.factory.engines), 1)
        self.assertIn("model_identity", report)
        self.assertEqual(report["studies"][0]["control_ready"], "fixture READY")

    def test_nonfinite_timing_cannot_complete_and_error_json_is_valid(self):
        def fault(engine, prompt, logprobs, result):
            if engine.engine != self.baseline and engine.count == 4:
                result["engine_decode_tokens_per_second"] = float("nan")
        self.factory.fault = fault
        with self.assertRaises(ValueError):
            self.run_bench()
        report = self.failed_report()
        value = report["studies"][0]["records"][0]["candidate"]["engine_decode_tokens_per_second"]
        self.assertEqual(value, {"invalid_nonfinite_value": "nan"})

    def test_binary_bank_and_model_changes_invalidate_measurement(self):
        for path, message, flags in ((self.optimized, "engine binary", ()),
                                    (self.heads, "head bank", ("--heads", str(self.heads))),
                                    (self.model / "model.safetensors", "model assets", ())):
            with self.subTest(path=path):
                self.factory = Factory(self.baseline)
                self.factory.close_fault = lambda engine: path.write_bytes(b"mutated " + path.read_bytes())
                self.output = self.root / (path.name + ".report.json")
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_bench(*flags)
                self.failed_report()

    def test_cleanup_failure_marks_study_and_report_failed(self):
        def failure(engine):
            raise RuntimeError("fixture close failed")
        self.factory.close_fault = failure
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            self.run_bench()
        report = self.failed_report()
        self.assertEqual(report["studies"][0]["status"], "failed")
        self.assertEqual(len(report["studies"][0]["cleanup_errors"]), 2)

    def test_midrun_checkpoint_contains_current_completed_pairs(self):
        checkpoints = []
        def observe(engine, prompt, logprobs, result):
            if engine.count >= 5:
                checkpoints.append(json.loads(self.output.read_text(encoding="utf-8")))
        self.factory.fault = observe
        self.run_bench()
        self.assertTrue(any(report["status"] == "running" and report["studies"][0]["records"]
                            for report in checkpoints))


class ProfileTest(unittest.TestCase):
    def test_mtp_statistics_require_complete_consistent_finite_contract(self):
        line = "[qwen36 MTP] turn bench1: bank_calls=1 bank_seconds=0.01 failures=0 forwards=32 tokens=63 accepted=8 drafts=16 verifies=8 ended_mid_verify=0"
        self.assertEqual(bench.mtp_statistics([line])["bench1"]["accepted"], 8)
        for lines in ([line, line], [line.replace("accepted=8", "accepted=17")],
                      [line.replace("bank_seconds=0.01", "bank_seconds=nan")],
                      [line.replace("bank_seconds=0.01", "bank_seconds=1e999")], [line + " ignored"],
                      [line.replace("ended_mid_verify=0", "ended_mid_verify=2")]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                bench.mtp_statistics(lines)

    def test_actual_protocol_layout_and_bad_counters(self):
        result = bench.parse_profile(b"PROF 1.5 12 64 0.01 0.02 0.3 0.4 0.5 32\n")
        self.assertEqual(result["prompt_tokens"], 12)
        self.assertEqual(result["completion_tokens"], 64)
        self.assertEqual(result["decode_seconds"], 1.5)
        self.assertEqual(result["forwards"], 32)
        self.assertEqual(result["moe_expert_seconds"], 0.3)
        self.assertEqual(result["output_head_seconds"], 0.5)
        self.assertIn("excludes other dense/GDN", result["phase_scope"]["moe_expert_seconds"])
        self.assertIn("draft-bank cost is not isolated", result["phase_scope"]["output_head_seconds"])
        self.assertNotIn("matmul_seconds", result)
        self.assertNotIn("head_seconds", result)
        for line in (b"PROF 1 2", b"PROF nan 12 64 0 0 0 0 0 1", b"PROF 1 -1 64 0 0 0 0 0 1",
                     b"PROF 1 12 513 0 0 0 0 0 1", b"PROF 1 12 64 0 0 0 0 0 -1"):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    bench.parse_profile(line)

    def test_metric_engine_counts_and_unavailable_prefill(self):
        engine = object.__new__(bench.MetricEngine)
        engine.phase_timers_enabled = False
        result = {"tokens": 64, "ttft_seconds": 0.2, "accept_seconds": 0.05}
        def turn(instance, *args, **kwargs):
            instance._observations["prompt_tokens"] = 12
            instance._observations["profiles"] = [bench.parse_profile(b"PROF 1.5 12 64 0 0 0 0 0 32")]
            return dict(result)
        with patch.object(bench.NativeEngine, "turn", turn):
            measured = engine.turn("fixture")
        self.assertIsNone(measured["prefill_seconds"])
        self.assertIsNone(measured["prefill_tokens_per_second"])
        self.assertFalse(measured["phase_seconds_available"])
        self.assertEqual(measured["prompt_tokens"], 12)
        self.assertAlmostEqual(measured["observed_prompt_to_first_data_seconds"], 0.15)
        self.assertIsNone(engine._observations)
        result["tokens"] = 63
        with patch.object(bench.NativeEngine, "turn", turn), self.assertRaisesRegex(ValueError, "counts differ"):
            engine.turn("fixture")


if __name__ == "__main__":
    unittest.main()

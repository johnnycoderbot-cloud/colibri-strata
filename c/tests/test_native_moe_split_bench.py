"""Model-free collector/real-wire fixtures; no GPU or throughput qualification."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch

from tools import native_decode_bench as driver
from tools import native_moe_split_bench as bench
from test_native_decode_bench import Pipe, Process, Sink


class WireSink(Sink):
    def __init__(self, process): super().__init__(); self.process=process
    def write(self, data):
        result=super().write(data)
        if data.startswith(b"SUBMIT "): self.process.answer(data)
        return result


class WireProcess(Process):
    def __init__(self, mode, scenario):
        super().__init__();self.mode=mode;self.scenario=scenario;self.count=0
        self.stdin=WireSink(self);self.stderr=Pipe(short=4096)
        name="Intel fixture device" if scenario.get("different_device") and mode!="off" else "llvmpipe (LLVM fixture)"
        self.stderr.feed(f"[VK] ready: {name}, compute qfam 0, memtype 1, fused gate+up\n".encode())
    def answer(self, request):
        self.count+=1;rid=request.split(b"\n",1)[0].split()[1].decode();candidate=self.mode!="off"
        lp=-.1-(self.scenario.get("lp_error",.001) if candidate else 0)
        frames=f"ACCEPT {rid}\n".encode()
        for index in range(8):
            token=index+1+(1 if candidate and self.scenario.get("wrong_ids") and index==0 else 0)
            frames+=f"DATA {rid} 1 {lp:.6f} 1 {token} {lp:.6f}\n".encode()+bytes([97+index])+b"\n"
        rate=16 if candidate else 8;seconds=.5 if candidate else 1
        frames+=f"DONE {rid} STAT 8 {rate} 75 1.2 5 1\nPROF {seconds} 5 8 .1 0 .2 .1 .1 8\n".encode()
        self.stdout.feed(frames)
        self.stderr.feed(f"[olmoe PILOT] turn {rid}: queued=0 completed=0 loads=0\n".encode())
        if not (candidate and self.scenario.get("missing_evidence")):
            row=f"[VK] tier olmoe turn: device {self.count*3} of {self.count*8} routed experts (37.5%; this turn 3 of 8) | replacement: LRU"
            if candidate:
                decisions=0 if self.scenario.get("no_decisions") else self.count*3
                row+=f" | split: {self.mode} {decisions} decisions, planned cold device {self.count*2} CPU {self.count*4}, deferred 1, calibration 1, predicted {self.count*.5:.3f} ms (live decode costs)"
            self.stderr.feed((row+"\n").encode())
        if self.count==1 and self.scenario.get("mutate_asset"):
            self.scenario["mutate_asset"].write_bytes(b"changed binary")
    def wait(self,timeout):
        result=super().wait(timeout);self.stderr.finish();return result
    def kill(self):
        super().kill();self.stderr.finish()


class WireEngine(bench.MoeMetricEngine):
    instances=[];scenario={}
    def __init__(self, engine, model, threads, env, wsl, **kwargs):
        self.ambient_selection={key:value for key,value in os.environ.items() if key.startswith(("COLI_VK","COLI_MOJO","CUDA_","VK_"))}
        self.fixture=WireProcess(env["COLI_VK_TIER_SPLIT"],self.scenario)
        with patch.object(driver.subprocess,"Popen",return_value=self.fixture):
            super().__init__(engine,model,threads,env,wsl,**kwargs)
        self.instances.append(self)
    def close(self):
        super().close()
        if self.scenario.get("close_failure"): raise RuntimeError("fixture close failure after process drain")


class CollectorTest(unittest.TestCase):
    def fixture(self, candidate="mojo"):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)
        engine=root/"olmoe";engine.write_bytes(b"fixture binary")
        model=root/"model";model.mkdir()
        (model/"config.json").write_text('{"model_type":"olmoe"}')
        (model/"tokenizer.json").write_text('{}');(model/"model.safetensors").write_bytes(b"fixture weights")
        source=root/"source";source.mkdir();(source/"olmoe.c").write_text('/* declared fixture source */')
        shaders=root/"shaders";shaders.mkdir();(shaders/"qmatmul.spv").write_bytes(b"fixture shader")
        library=root/"mojo.so";library.write_bytes(b"fixture Mojo library")
        prompts=root/"prompts.json";prompts.write_text('["first","second","third"]')
        output=root/"report.json"
        args=["--engine",str(engine),"--model",str(model),"--prompts",str(prompts),"--output",str(output),
              "--source-root",str(source),"--shaders",str(shaders),"--candidate",candidate,"--tokens","8","--repeats","3"]
        if candidate=="mojo":args += ["--mojo-library",str(library)]
        return root,engine,output,args
    def run_collector(self, scenario=None, candidate="mojo"):
        root,engine,output,args=self.fixture(candidate)
        WireEngine.instances=[];WireEngine.scenario=dict(scenario or {})
        if WireEngine.scenario.get("mutate_asset"):WireEngine.scenario["mutate_asset"]=engine
        with contextlib.redirect_stdout(io.StringIO()):
            if scenario:
                with self.assertRaises((RuntimeError,ValueError)):bench.main(args,engine_factory=WireEngine)
            else:bench.main(args,engine_factory=WireEngine)
        self.assertTrue(all(item.fixture.poll() is not None for item in WireEngine.instances))
        self.assertTrue(all(not thread.is_alive() for item in WireEngine.instances for thread in item.readers))
        return json.loads(output.read_text()),list(WireEngine.instances)

    def test_paired_real_wire_collector_numeric_tolerance_and_scope(self):
        report,engines=self.run_collector()
        self.assertEqual(report["status"],"complete")
        self.assertEqual(len(report["records"]),9)
        self.assertEqual(report["summary"]["median_paired_whole_request_speed_ratio"],2)
        self.assertTrue(report["assets_verified_before_and_after"])
        self.assertTrue(report["fake_engine_factory"])
        self.assertFalse(report["physical_gpu_execution_qualified"] or report["gpu_speed_claim_allowed"])
        self.assertIn("software Vulkan",report["execution_scope"])
        self.assertTrue(report["arms"]["candidate"]["device"]["software_device"])
        first=report["records"][0]
        self.assertAlmostEqual(first["numeric_comparison"]["max_absolute_logprob_error"],.001)
        self.assertFalse(first["numeric_comparison"]["exact_numeric_frames"])
        cumulative=first["candidate"]["native_tier_statistics"]["split_cumulative"]
        self.assertEqual(cumulative["decisions"],9)
        self.assertEqual(first["candidate"]["native_tier_statistics"]["split_since_previous_turn"]["decisions"],3)
        self.assertEqual(report["records"][0]["order"],["control","candidate"])
        self.assertEqual(report["records"][1]["order"],["candidate","control"])
        for engine in engines:
            for key,value in {"PILOT":"0","IDOT":"0","FUSED3":"0","COLI_VK_TIER_SYNC":"1","COLI_LOOKUP":"0","COLI_KV_PREFIX":"0"}.items():
                self.assertEqual(engine.launch_settings[key],value)
        differences={key for key in set(engines[0].launch_settings)|set(engines[1].launch_settings)
                     if engines[0].launch_settings.get(key)!=engines[1].launch_settings.get(key)}
        self.assertEqual(differences,{"COLI_VK_TIER_SPLIT","COLI_MOJO_DISPATCH"})

    def test_native_candidate_uses_same_protocol_without_plugin(self):
        report,engines=self.run_collector(candidate="native")
        self.assertEqual(report["status"],"complete")
        self.assertNotIn("COLI_MOJO_DISPATCH",engines[1].launch_settings)
        self.assertEqual(report["records"][0]["candidate"]["native_tier_statistics"]["split_mode"],"native")

    def test_numeric_execution_device_asset_and_cleanup_failures_are_persisted(self):
        for scenario in ({"wrong_ids":True},{"lp_error":.03},{"missing_evidence":True},{"no_decisions":True},
                         {"different_device":True},{"mutate_asset":True},{"close_failure":True}):
            with self.subTest(scenario=scenario):
                report,_=self.run_collector(scenario)
                self.assertEqual(report["status"],"failed")
                self.assertTrue(report["correctness"])
                self.assertTrue(report.get("error") or report.get("qualification_errors") or report.get("cleanup_errors"))

    def test_ambient_selection_overrides_and_secret_values_are_not_forwarded(self):
        ambient={"COLI_MOJO_DISPATCH":"private-plugin-value","COLI_VK_TIER_GB":"private-tier-value",
                 "VK_ICD_FILENAMES":"private-driver-value","CUDA_VISIBLE_DEVICES":"private-cuda-value"}
        with patch.dict(os.environ,ambient):
            report,engines=self.run_collector()
            self.assertTrue(all(not engine.ambient_selection for engine in engines))
            self.assertEqual({key:os.environ[key] for key in ambient},ambient)
        encoded=json.dumps(report)
        for value in ambient.values():self.assertNotIn(value,encoded)

    def test_argument_bounds_refuse_before_launch_or_report_overwrite(self):
        _,_,output,args=self.fixture()
        for extension in (["--tokens","7"],["--tokens","65"],["--repeats","2"],["--repeats","21"],
                          ["--tier-gb","nan"],["--max-logprob-error",".11"],["--cache-per-layer","9"],
                          ["--request-timeout","inf"]):
            with self.subTest(extension=extension),self.assertRaises(ValueError):bench.main(args+extension,engine_factory=WireEngine)
        self.assertFalse(output.exists())
        no_plugin=args[:-2]
        with self.assertRaisesRegex(ValueError,"mojo-library"):bench.main(no_plugin,engine_factory=WireEngine)
        output.write_text("keep")
        with self.assertRaises(FileExistsError):bench.main(args,engine_factory=WireEngine)
        self.assertEqual(output.read_text(),"keep")

    def test_split_counter_deltas_and_malformed_telemetry(self):
        base="[VK] tier olmoe turn: device 5 of 10 routed experts (fixture)"
        tail=" | split: mojo 2 decisions, planned cold device 2 CPU 3, deferred 1, calibration 1, predicted 1.500 ms (live decode costs)"
        stats=bench.tier_statistics([base+tail,base+tail.replace("2 decisions","5 decisions").replace("1.500","2.500")])
        self.assertEqual(stats[1]["split_since_previous_turn"]["decisions"],3)
        self.assertEqual(stats[1]["split_since_previous_turn"]["predicted_ms"],1)
        for rows in ([base+tail,base+tail.replace("2 decisions","1 decisions")],
                     [base.replace("5 of 10","11 of 10")], [base+" | split: mojo broken"],
                     [base+tail.replace("1.500","nan")]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):bench.tier_statistics(rows)
        with self.assertRaises(RuntimeError):bench.device_evidence(["[VK] no usable device; running on CPU"])


if __name__=="__main__":unittest.main()

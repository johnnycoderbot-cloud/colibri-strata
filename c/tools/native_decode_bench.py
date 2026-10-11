#!/usr/bin/env python3
"""Persistent native-engine paired measurements and real hidden-trace capture.

No model downloads, synthetic perfect proposals, or accelerator substitution.
The optional WSL launcher passes argument arrays directly, never shell text.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import statistics
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
READY=b"\x01\x01READY\x01\x01"
MAX_FRAME=4<<20
MAX_LINE=64<<10
MAX_STDERR=1<<20
MAX_PROMPT=1<<20

def bounded_int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value

def bounded_number(value, low, high, name, positive=False):
    if type(value) not in (int,float) or not low <= value <= high or not math.isfinite(value) or (positive and value==low):
        raise ValueError(f"{name} must be finite within its bounds")
    return value

def read_exact(stream,count):
    chunks=[]
    while count:
        chunk=stream.read(count)
        if not chunk: raise ValueError("truncated native frame body")
        if len(chunk)>count: raise ValueError("native reader exceeded requested byte count")
        chunks.append(chunk); count-=len(chunk)
    return b"".join(chunks)

class FrameQueue(queue.Queue):
    """Tuple frames stay compatible; direct frontier callers see parse errors."""
    def __init__(self):
        super().__init__(maxsize=16); self.failure=None
    def get(self,*args,**kwargs):
        if self.failure is not None: raise self.failure
        result=super().get(*args,**kwargs)
        if self.failure is not None: raise self.failure
        return result
    def finish(self):
        with self.mutex:
            self.queue.clear(); self.queue.append((b"",None))
            self.not_empty.notify_all(); self.not_full.notify_all()

def linux_path(path):
    value=Path(path).resolve().as_posix()
    if len(value)>2 and value[1:3]==":/":
        return "/mnt/"+value[0].lower()+"/"+value[3:]
    return value

class NativeEngine:
    def __init__(self, engine, model, threads=8, env=None, wsl=None, *, startup_timeout=180, close_timeout=20,
                 cache_per_layer=1, expert_bits=8):
        bounded_int(threads,1,64,"threads")
        bounded_int(cache_per_layer,1,4096,"cache_per_layer");bounded_int(expert_bits,2,16,"expert_bits")
        bounded_number(startup_timeout,0,3600,"startup_timeout",True)
        bounded_number(close_timeout,0,60,"close_timeout")
        if wsl is not None and (not isinstance(wsl,str) or not wsl or len(wsl)>128 or any(ord(c)<32 for c in wsl)):
            raise ValueError("invalid WSL distribution")
        self.frames=FrameQueue(); self.stderr=[]; self.count=0
        self._stop=threading.Event(); self._closed=False; self._close_lock=threading.Lock()
        self._turn_lock=threading.Lock(); self._close_timeout=close_timeout
        self._stderr_bytes=0; self.stderr_truncated_bytes=0; self.ready_line=None
        values={"SNAP":linux_path(model) if wsl else str(Path(model).resolve()),
                "SERVE":"1","OMP_NUM_THREADS":str(threads),"COLI_NO_OMP_TUNE":"1",
                "COLI_LOOKUP":"0","USAGE_SAVE":"0","CTX":"2048","KV_SLOTS":"1",
                "COLI_DENSE_BITS":"8","COLI_DENSE_I8":"1","COLI_DENSE_IDOT":"1"}
        for key,value in (env or {}).items():
            if not isinstance(key,str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",key) or "\0" in str(value):
                raise ValueError("invalid environment override")
            values[key]=str(value)
        self.launch_settings={key:value for key,value in values.items() if not re.search(r"TOKEN|SECRET|PASSWORD|AUTH|API_KEY",key,re.I)}
        if wsl:
            args=["wsl.exe","-d",wsl,"--cd",linux_path(ROOT),"--","env"]
            args += [key+"="+str(value) for key,value in values.items()]
            args += [linux_path(engine),str(cache_per_layer),str(expert_bits)]
            child_env=None
        else:
            args=[str(Path(engine).resolve()),str(cache_per_layer),str(expert_bits)]
            child_env=os.environ.copy(); child_env.update(values)
        self.process=subprocess.Popen(args,cwd=ROOT,env=child_env,stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.readers=[threading.Thread(target=self._read,daemon=True),threading.Thread(target=self._errors,daemon=True)]
        for reader in self.readers: reader.start()
        deadline=time.monotonic()+startup_timeout
        try:
            while True:
                line,_=self._next(deadline)
                if line.startswith(READY):
                    self.ready_line=line.decode("utf-8","replace").strip(); break
                if not line: raise RuntimeError("native engine exited before READY: "+"".join(self.stderr)[-3000:])
        except BaseException as error:
            try: self.close()
            except Exception as cleanup:
                if hasattr(error,"add_note"): error.add_note(f"cleanup failed: {cleanup}")
            raise

    def _put(self,item):
        while not self._stop.is_set():
            try: self.frames.put(item,timeout=0.05); return
            except queue.Full: pass

    def _next(self,deadline):
        if self._closed: raise RuntimeError("native engine is closed")
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError("native engine response timed out")
        try: return self.frames.get(timeout=remaining)
        except queue.Empty as error: raise TimeoutError("native engine response timed out") from error

    def _read(self):
        try:
            while True:  # drain pipes through shutdown; only queue publication stops
                line=self.process.stdout.readline(MAX_LINE+1)
                if not line: break
                if len(line)>MAX_LINE or not line.endswith(b"\n"): raise ValueError("oversized or truncated native header")
                body=None
                if line.startswith((b"DATA ",b"ECHO ")):
                    fields=line.split()
                    if len(fields)<3 or not fields[2].isdigit(): raise ValueError("invalid native frame length")
                    count=int(fields[2])
                    if count>MAX_FRAME: raise ValueError("invalid native frame length")
                    body=read_exact(self.process.stdout,count)
                    if read_exact(self.process.stdout,1)!=b"\n": raise ValueError("invalid native frame delimiter")
                self._put((line,body))
        except BaseException as error:
            if not self._stop.is_set(): self.frames.failure=error
        finally:
            self._put((b"",None))

    def _errors(self):
        while True:
            raw=self.process.stderr.readline(MAX_LINE)
            if not raw: return
            line=raw.decode("utf-8","replace"); self.stderr.append(line)
            self._stderr_bytes+=len(line.encode("utf-8"))
            while (self._stderr_bytes>MAX_STDERR or len(self.stderr)>2048) and self.stderr:
                lost=len(self.stderr.pop(0).encode("utf-8")); self._stderr_bytes-=lost
                self.stderr_truncated_bytes+=lost

    def turn(self,prompt,max_tokens=64,logprobs=False,timeout=120,extra="",temperature=0,top_p=1,cancel_after=None):
        bounded_int(max_tokens,1,512,"max_tokens"); bounded_number(timeout,0,3600,"timeout",True)
        bounded_number(temperature,0,5,"temperature"); bounded_number(top_p,0,1,"top_p",True)
        if type(logprobs) is not bool: raise ValueError("logprobs must be a boolean")
        if cancel_after is not None: bounded_int(cancel_after,1,max_tokens,"cancel_after")
        if not isinstance(prompt,str) or not prompt or len(prompt)>MAX_PROMPT or len(prompt.encode("utf-8","strict"))>MAX_PROMPT:
            raise ValueError("prompt must be bounded nonempty UTF-8")
        if not isinstance(extra,str) or len(extra)>4096 or any(not 32<=ord(c)<=126 for c in extra):
            raise ValueError("invalid native extension")
        seen={"logprobs"} if logprobs else set()
        for item in extra.split():
            if not re.fullmatch(r"[a-z][a-z0-9_]*=[!-~]+",item): raise ValueError("extensions must be ASCII key=value tokens")
            key=item.split("=",1)[0]
            if key in seen or len(seen)>=16: raise ValueError("duplicate or excessive native extension")
            seen.add(key)
            if key in ("pin","gbytes_before_ext","logprobs"):
                raw=item.split("=",1)[1]
                if not raw.isdigit(): raise ValueError("invalid numeric native extension")
                bounded_int(int(raw),0,20 if key=="logprobs" else 1,key)
        with self._turn_lock:
            try: return self._turn(prompt,max_tokens,logprobs,timeout,extra,temperature,top_p,cancel_after)
            except BaseException as error:
                try:
                    if self.process.poll() is None:
                        self.process.stdin.write(f"CANCEL bench{self.count}\n".encode()); self.process.stdin.flush()
                except (OSError,ValueError): pass
                try: self.close()
                except Exception as cleanup:
                    if hasattr(error,"add_note"): error.add_note(f"cleanup failed: {cleanup}")
                raise

    def _turn(self,prompt,max_tokens,logprobs,timeout,extra,temperature,top_p,cancel_after):
        if self._closed: raise RuntimeError("native engine is closed")
        self.count+=1; request_id="bench"+str(self.count); payload=prompt.encode("utf-8")
        extension=" logprobs=1" if logprobs else ""
        if "\n" in extra or "\r" in extra: raise ValueError("invalid native extension")
        if extra: extension+=" "+extra
        start=time.perf_counter(); deadline=time.monotonic()+timeout
        first=None; accepted=None; bodies=[]; ids=[]; headers=[]; numeric=[]
        self.process.stdin.write(f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} {temperature} {top_p}{extension}\n".encode()+payload+b"\n")
        self.process.stdin.flush()
        while True:
            line,body=self._next(deadline)
            if not line: raise RuntimeError("native engine died: "+"".join(self.stderr)[-3000:])
            fields=line.split()
            if not fields: continue
            if len(fields)>1 and fields[1].decode()!=request_id: continue
            if fields[0]==b"ACCEPT": accepted=time.perf_counter()-start
            elif fields[0]==b"DATA":
                if first is None: first=time.perf_counter()-start
                bodies.append(body)
                if cancel_after is not None and len(bodies)==cancel_after:
                    self.process.stdin.write(f"CANCEL {request_id}\n".encode()); self.process.stdin.flush()
                if logprobs:
                    if len(fields)<7: raise ValueError("missing native token evidence")
                    # DATA carries chosen LP and top-k pairs, with no position
                    # or explicit chosen token ID. At temperature=0, top1 is
                    # the chosen greedy token; sampled IDs cannot be inferred.
                    lp=float(fields[3]); top_count=bounded_int(int(fields[4]),1,20,"top count")
                    if not math.isfinite(lp): raise ValueError("nonfinite native logprob")
                    if len(fields)!=5+2*top_count: raise ValueError("invalid native top-token pairs")
                    top=[]
                    for i in range(top_count):
                        token=bounded_int(int(fields[5+2*i]),0,2**31-1,"token id")
                        top_lp=float(fields[6+2*i])
                        if not math.isfinite(top_lp): raise ValueError("nonfinite native top logprob")
                        top.append({"token_id":token,"logprob":top_lp})
                    if len({item["token_id"] for item in top})!=top_count: raise ValueError("duplicate native top token")
                    token=top[0]["token_id"] if temperature==0 else None
                    if token is not None: ids.append(token)
                    headers.append(line.decode().strip())
                    numeric.append({"emitted_index":len(numeric),"token_id":token,
                                    "token_id_source":"top1_greedy" if token is not None else None,
                                    "logprob":lp,"top_tokens":top,"body_sha256":hashlib.sha256(body).hexdigest()})
            elif fields[0]==b"ERROR":
                if cancel_after is not None and fields[2:]==[b"CANCELLED"]:
                    text=b"".join(bodies)
                    return {"cancelled":True,"request_id":request_id,"token_ids":ids,"numeric_frames":headers,"numeric_evidence":numeric,"tokens":len(numeric) if logprobs else None,"token_count_verified":logprobs,"token_ids_verified":logprobs and temperature==0,"text":text.decode("utf-8","replace"),"output_sha256":hashlib.sha256(text).hexdigest()}
                raise RuntimeError(line.decode())
            elif fields[0]==b"DONE":
                elapsed=time.perf_counter()-start
                if len(fields)<5 or fields[2]!=b"STAT": raise ValueError("missing native timing")
                count=bounded_int(int(fields[3]),0,max_tokens,"completed tokens")
                rate=bounded_number(float(fields[4]),0,1e9,"decode rate",count>0)
                if logprobs and count!=len(numeric): raise ValueError("native token evidence/count mismatch")
                text=b"".join(bodies)
                return {"request_id":request_id,"tokens":count,"engine_decode_tokens_per_second":rate,
                        "wall_seconds":elapsed,"ttft_seconds":first,"accept_seconds":accepted,
                        "output_sha256":hashlib.sha256(text).hexdigest(),"token_ids":ids,
                        "token_ids_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest() if logprobs else None,
                        "text":text.decode("utf-8","replace"),"numeric_frames":headers,"numeric_evidence":numeric,
                        "token_ids_verified":logprobs and temperature==0}

    def close(self):
        with self._close_lock:
            if self._closed: return
            self._closed=True; self._stop.set()
            self.frames.finish()
            if self.process.poll() is None:
                try: self.process.stdin.close(); self.process.wait(timeout=self._close_timeout)
                except (OSError,subprocess.TimeoutExpired): self.process.kill(); self.process.wait(timeout=10)
            for reader in self.readers:
                if reader.ident is not None and reader is not threading.current_thread(): reader.join(timeout=3)
            for stream in (self.process.stdin,self.process.stdout,self.process.stderr):
                if stream: stream.close()

def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk:=stream.read(65536): value.update(chunk)
    return value.hexdigest()

def identity(engine,model):
    result={"binary":str(Path(engine).resolve()),"model":str(Path(model).resolve()),
            "launch_contract":"SUBMIT/DATA/DONE; positional cap=1,bits=8",
            "accelerator_execution_verified":False,"weights_hashed":False,"metadata":{}}
    if Path(engine).is_file(): result["binary_sha256"]=file_sha(engine)
    for name in ("config.json","qwen36_meta.json","model.safetensors.index.json"):
        path=Path(model)/name
        if path.is_file() and not path.is_symlink() and path.stat().st_size<=1<<20:
            result["metadata"][name]={"sha256":file_sha(path),"bytes":path.stat().st_size}
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",required=True); parser.add_argument("--feature",required=True)
    parser.add_argument("--model",required=True); parser.add_argument("--prompts",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--heads")
    parser.add_argument("--head-input",choices=("hidden","next-embedding"),default="hidden")
    parser.add_argument("--wsl"); parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--tokens",type=int,default=64); parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--lookup",action="store_true")
    args=parser.parse_args(argv)
    if not 1<=args.threads<=64 or not 1<=args.tokens<=512 or not 5<=args.repeats<=100: parser.error("invalid benchmark bounds")
    output=Path(args.output)
    if output.exists(): parser.error("output must be new")
    output.parent.mkdir(parents=True,exist_ok=True)
    prompts=json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts,list) or not prompts or not all(isinstance(p,str) for p in prompts): parser.error("prompts must be a nonempty JSON string list")
    common={"COLI_LOOKUP":"1" if args.lookup else "0","COLI_CUDA":"0","COLI_VULKAN":"0",
            "COLI_STRATA_HEADS":"","COLI_STRATA_TRACE":"","COLI_STRATA_INPUT":"",
            "COLI_SPEC_GATE":"1","QWEN_HEAD_BATCH":"1","QWEN_EXPERT_BATCH":"1"}
    feature_env=dict(common)
    if args.heads: feature_env["COLI_STRATA_HEADS"]=linux_path(args.heads) if args.wsl else str(Path(args.heads).resolve())
    if args.heads: feature_env["COLI_STRATA_INPUT"]=args.head_input
    baseline=feature=None
    records=[]; correctness=[]; failure=None
    report={"schema":"colibri.native-paired-benchmark/v1","status":"running","threads":args.threads,
            "repeats":args.repeats,"records":records,"correctness":correctness}
    try:
        report["baseline_identity"]=identity(args.baseline,args.model)
        report["feature_identity"]=identity(args.feature,args.model)
        report["baseline_binary_sha256"]=report["baseline_identity"].get("binary_sha256")
        report["feature_binary_sha256"]=report["feature_identity"].get("binary_sha256")
        report["prompt_sha256"]=file_sha(args.prompts)
        report["head_sha256"]=file_sha(args.heads) if args.heads else None
        baseline=NativeEngine(args.baseline,args.model,args.threads,common,args.wsl)
        feature=NativeEngine(args.feature,args.model,args.threads,feature_env,args.wsl)
        for index,prompt in enumerate(prompts):
            a=baseline.turn(prompt,args.tokens,True); b=feature.turn(prompt,args.tokens,True)
            equal=(a["token_ids"]==b["token_ids"] and a["output_sha256"]==b["output_sha256"] and
                   [x.split(" ",2)[2] for x in a["numeric_frames"]]==[x.split(" ",2)[2] for x in b["numeric_frames"]])
            correctness.append({"prompt":index,"equal":equal,"baseline":a,"feature":b})
            if not equal: raise RuntimeError(f"native correctness mismatch for prompt {index}")
            baseline.turn(prompt,args.tokens); feature.turn(prompt,args.tokens)
            for repeat in range(args.repeats):
                arms=(("baseline",baseline),("feature",feature)) if repeat%2==0 else (("feature",feature),("baseline",baseline))
                paired={name:engine.turn(prompt,args.tokens) for name,engine in arms}
                records.append({"prompt":index,"repeat":repeat,"order":[name for name,_ in arms],**paired})
                if paired["baseline"]["output_sha256"]!=paired["feature"]["output_sha256"] or paired["baseline"]["tokens"]!=paired["feature"]["tokens"]:
                    raise RuntimeError("paired native output differs")
                print(json.dumps({"prompt":index,"repeat":repeat,"baseline_tps":paired["baseline"]["engine_decode_tokens_per_second"],"feature_tps":paired["feature"]["engine_decode_tokens_per_second"]}),flush=True)
        ratios=[bounded_number(r["feature"]["engine_decode_tokens_per_second"],0,1e9,"feature rate",True)/
                bounded_number(r["baseline"]["engine_decode_tokens_per_second"],0,1e9,"baseline rate",True) for r in records]
        report["median_paired_decode_ratio"]=statistics.median(ratios); report["status"]="complete"
    except BaseException as error:
        failure=error; report["status"]="failed"
        report["error"]={"type":type(error).__name__,"message":str(error)}
    finally:
        for name,engine in (("baseline",baseline),("feature",feature)):
            if engine:
                try: engine.close()
                except BaseException as error:
                    failure=failure or error; report["status"]="failed"
                    report.setdefault("cleanup_errors",[]).append({"arm":name,"type":type(error).__name__,"message":str(error)})
                report[name+"_stderr"]=list(engine.stderr)
                report[name+"_stderr_truncated_bytes"]=getattr(engine,"stderr_truncated_bytes",0)
                report[name+"_ready"]=getattr(engine,"ready_line",None)
                report[name+"_launch_settings"]=getattr(engine,"launch_settings",{})
    with output.open("x",encoding="utf-8") as stream: json.dump(report,stream,indent=2); stream.write("\n")
    if failure is not None: raise failure

if __name__=="__main__": main()

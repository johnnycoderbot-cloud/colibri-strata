"""Standard-library launcher for native Mojo/MAX C inference; no SDK bindings."""
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def run(args):
    if sys.platform == "win32":
        raise RuntimeError("Modular requires Linux/macOS; run this command inside WSL or on a native supported host")
    binary = Path(args.native or os.environ.get("COLI_MOJO_NATIVE", Path(__file__).resolve().parent / "mojo_runtime"))
    if not binary.is_file():
        raise RuntimeError("native Mojo backend is not built; run python c/tools/build_native_mojo.py in the SDK environment")
    if type(args.timeout) not in (float, int) or not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        raise ValueError("native timeout must be finite in (0,3600]")
    command = [str(binary.resolve())]
    if args.mojo_cmd == "probe":
        command += ["--probe"]
    elif args.mojo_cmd == "run":
        if not re.fullmatch(r"CPU|ACCELERATOR\.(?:0|[1-9][0-9]{0,2}|1000)", args.device):
            raise ValueError("device must be CPU or exact ACCELERATOR.id; NPU/TPU/AIPU need actual plugins")
        if type(args.repeat) is not int or not 1 <= args.repeat <= 10000:
            raise ValueError("repeat must be in [1,10000]")
        command += ["--model", str(Path(args.model).resolve()), "--inputs", str(Path(args.inputs).resolve()),
                    "--output", str(Path(args.output).resolve()), "--device", args.device,
                    "--repeat", str(args.repeat)]
        weights = getattr(args, "weights", [])
        if not isinstance(weights, list) or len(weights) > 32:
            raise ValueError("at most 32 safetensor shards are supported")
        for path in weights:
            command += ["--weights", str(Path(path).resolve())]
    else:
        raise ValueError("unknown Mojo command")
    # subprocess.run owns the child and kills/waits on timeout. No shell or
    # Python tensor/model work occurs in this launcher.
    return subprocess.run(command, timeout=args.timeout, check=False).returncode

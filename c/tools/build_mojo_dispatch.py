#!/usr/bin/env python3
"""Compile the native Mojo split planner; Python is build tooling only."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler", type=Path)
    p.add_argument("--sdk-root", type=Path,
                   help="matching SDK's modular directory containing lib/mojo/std.mojoc")
    p.add_argument("--output", type=Path)
    p.add_argument("--config-dir", type=Path)
    args = p.parse_args(argv)
    if sys.platform not in ("linux", "darwin"):
        p.error("native Mojo dispatch requires Linux/macOS; build and run inside WSL on Windows")
    if args.compiler is None or args.sdk_root is None:
        try:
            dist = importlib.metadata.distribution("mojo-compiler")
        except importlib.metadata.PackageNotFoundError:
            p.error("install official mojo-compiler==1.1.0 or supply matching --compiler and --sdk-root")
        if dist.version != "1.1.0":
            p.error("qualified compiler is Mojo1.1.0; supply explicit paths to qualify another SDK")
        args.compiler = args.compiler or Path(dist.locate_file("modular/bin/mojo"))
        args.sdk_root = args.sdk_root or Path(dist.locate_file("modular"))
    compiler = args.compiler.resolve()
    sdk = args.sdk_root.resolve()
    if not compiler.is_file() or not (sdk / "lib/mojo/std.mojoc").is_file():
        p.error("missing real Mojo compiler or its matching native standard library")
    if any(c in str(sdk) for c in "\r\n"):
        p.error("invalid SDK resource path")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = (args.output or ROOT / ("libcoli_mojo_dispatch" + suffix)).resolve()
    if output == (ROOT / "mojo_split_dispatch.mojo").resolve():
        p.error("output would overwrite the Mojo source")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = (args.config_dir or Path(str(output) + ".sdk")).resolve()
    cfg = config / "modular.cfg"
    content = ("[max]\npackage_root = " + str(sdk) +
               "\n[mojo-max]\npackage_root = " + str(sdk) +
               "\nimport_path = " + str(sdk / "lib/mojo") + "\n")
    config.mkdir(parents=True, exist_ok=True)
    if cfg.exists() and cfg.read_text(encoding="utf-8") != content:
        p.error("SDK configuration differs; select a fresh --config-dir")
    if not cfg.exists():
        cfg.write_text(content, encoding="utf-8")
    env = dict(os.environ)
    env["MODULAR_HOME"] = str(config)
    # Keep a previous qualified binary intact if compilation/linking fails.
    with tempfile.TemporaryDirectory(prefix=".mojo-dispatch-build-", dir=output.parent) as temp:
        candidate = Path(temp) / output.name
        subprocess.run([str(compiler), "build", str(ROOT / "mojo_split_dispatch.mojo"),
                        "--emit", "shared-lib", "--fp-mode", "contract=off",
                        "-Xlinker", "-rpath", "-Xlinker", str(sdk / "lib"),
                        "-o", str(candidate)], check=True, env=env)
        if not candidate.is_file():
            raise RuntimeError("Mojo compiler returned without producing a shared library")
        os.replace(candidate, output)
    print(output)
    return 0


if __name__ == "__main__":
    main()

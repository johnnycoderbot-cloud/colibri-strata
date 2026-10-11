#!/usr/bin/env python3
"""Build the C Mojo/MAX backend using the real installed native SDK."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include", type=Path)
    p.add_argument("--lib", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "mojo_runtime")
    p.add_argument("--cc", default=os.environ.get("CC", "cc"))
    p.add_argument("--test", action="store_true")
    p.add_argument("--sanitize", action="store_true")
    p.add_argument("--config-dir", type=Path)
    args = p.parse_args(argv)
    if sys.platform == "win32":
        p.error("Modular SDK currently requires native Linux/macOS or WSL; build inside that environment")
    if args.include is None or args.lib is None:
        try:
            dist = importlib.metadata.distribution("max-core")
        except importlib.metadata.PackageNotFoundError:
            p.error("install the matching official MAX SDK or supply --include and --lib")
        if dist.version != "26.6.0":
            p.error("qualified SDK is MAX26.6.0; supply explicit SDK paths to qualify another version")
        args.include = args.include or Path(dist.locate_file("max/include"))
        args.lib = args.lib or Path(dist.locate_file("modular/lib"))
    include, library = args.include.resolve(), args.lib.resolve()
    if not (include / "max/c/model.h").is_file():
        p.error("missing real SDK C headers")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = (args.config_dir or output.with_suffix(".sdk")).resolve()
    content = "[max]\npackage_root = " + str(library.parent) + "\n[mojo-max]\npackage_root = " + str(library.parent) + "\nimport_path = " + str(library.parent / "lib/mojo") + "\n"
    if any(c in str(library.parent) for c in "\r\n"):
        p.error("invalid SDK path")
    config.mkdir(parents=True, exist_ok=True)
    cfg = config / "modular.cfg"
    if cfg.exists() and cfg.read_text(encoding="utf-8") != content:
        p.error("SDK config differs; select a fresh --config-dir")
    if not cfg.exists():
        cfg.write_text(content, encoding="utf-8")
    main_source = "tests/test_native_mojo_backend.c" if args.test else "mojo_runtime_main.c"
    flags = ["-std=c17", "-O2", "-Wall", "-Wextra", "-Wno-misleading-indentation", "-Wno-unused-function"]
    if args.sanitize:
        flags += ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    subprocess.run([args.cc, *flags, "-DCOLI_MOJO_CONFIG_DIR=" + json.dumps(str(config), ensure_ascii=False),
                    "-I" + str(include), str(ROOT / main_source),
                    str(ROOT / "native_mojo_backend.c"), "-L" + str(library),
                    "-Wl,-rpath," + str(library), "-lmax", "-lm", "-o", str(output)],
                   check=True)
    print(output)


if __name__ == "__main__":
    main()

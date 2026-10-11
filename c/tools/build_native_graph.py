#!/usr/bin/env python3
"""Build the C graph engine; model SDKs are loaded natively at execution time."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("native_graph_main.c", "native_graph_backend.c", "native_graph_manifest.c", "native_media.c")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ("native_graph.exe" if os.name == "nt" else "native_graph"))
    parser.add_argument("--cc")
    parser.add_argument("--vcvars", type=Path)
    parser.add_argument("--test", choices=("media", "backend", "manifest"), help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="also compile and run SDK-free C contract checks")
    parser.add_argument("--sanitize", action="store_true", help="enable GCC/Clang address and undefined-behavior sanitizers")
    args = parser.parse_args(argv)
    if args.sanitize and os.name == "nt":
        parser.error("--sanitize requires a GCC/Clang sanitizer host")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = SOURCES
    if args.test:
        sources = {"media": ("tests/test_native_media.c", "native_media.c"),
                   "backend": ("tests/test_native_graph_backend.c",),
                   "manifest": ("tests/test_native_graph_manifest.c", "native_graph_manifest.c")}[args.test]
    if os.name != "nt":
        compiler = args.cc or os.environ.get("CC") or "cc"
        subprocess.run([compiler, "-std=c17", "-O2", "-Wall", "-Wextra", "-Wno-unused-function",
                        "-Wno-misleading-indentation",
                        *(["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else []),
                        *(str(ROOT / name) for name in sources),
                        "-o", str(output), "-lm", *([] if sys.platform == "darwin" else ["-ldl"])], check=True)
    else:
        vcvars = args.vcvars
        if vcvars is None:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            if not vswhere.is_file():
                raise RuntimeError("native C compiler unavailable; supply --vcvars VC/Auxiliary/Build/vcvars64.bat")
            install = subprocess.check_output([str(vswhere), "-latest", "-products", "*",
                         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
            vcvars = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
        if not vcvars.is_file() or any(c in str(vcvars) for c in '"%\r\n'):
            raise ValueError("invalid native compiler setup path")
        # Shell text only activates the trusted compiler environment; source/output
        # paths go in a response file, never interpolated into shell commands.
        # Windows environment keys are case-insensitive, but dict is not. A
        # stale inherited Path must not shadow vcvars' updated PATH value.
        env = {key.upper(): value for key, value in os.environ.items()}
        with tempfile.TemporaryDirectory(prefix="colibri-compiler-env-") as setup:
            batch = Path(setup) / "environment.cmd"
            batch.write_text(f'@call "{vcvars}" >nul\n@if errorlevel 1 exit /b 1\n@set\n', encoding="utf-8")
            for line in subprocess.check_output(["cmd.exe", "/d", "/c", str(batch)], text=True).splitlines():
                if "=" in line and not line.startswith("="):
                    key, value = line.split("=", 1)
                    env[key.upper()] = value
        with tempfile.TemporaryDirectory(prefix="colibri-native-c-") as temporary:
            work = Path(temporary)
            flags = ["/nologo", "/std:c17", "/O2", "/W4", "/D_CRT_SECURE_NO_WARNINGS", "/TC",
                     "/Fe" + str(output), "/Fo" + work.as_posix() + "/",
                     *(str(ROOT / name) for name in sources), "/link", "kernel32.lib", "shell32.lib"]
            response = work / "build.rsp"
            if any('"' in flag or "\n" in flag or "\r" in flag for flag in flags):
                raise ValueError("invalid compiler argument")
            response.write_text(" ".join('"' + flag + '"' for flag in flags) + "\n", encoding="utf-8-sig")
            compiler = args.cc or shutil.which("cl.exe", path=env.get("PATH"))
            if not compiler:
                raise RuntimeError("compiler setup did not provide cl.exe")
            subprocess.run([compiler, "@" + str(response)], env=env, cwd=work, check=True)
    # A redirected Windows console can still advertise a legacy code page.
    # Reporting the completed build must not fail on a valid Unicode path.
    encoding = sys.stdout.encoding or "utf-8"
    print(str(output).encode(encoding, "backslashreplace").decode(encoding))
    if args.check:
        with tempfile.TemporaryDirectory(prefix="colibri-native-contracts-") as tmp:
            for name in ("media", "backend", "manifest"):
                binary = Path(tmp) / (name + (".exe" if os.name == "nt" else ""))
                options = ["--test", name, "--output", str(binary)]
                if args.cc:
                    options += ["--cc", args.cc]
                if args.vcvars:
                    options += ["--vcvars", str(args.vcvars)]
                if args.sanitize:
                    options += ["--sanitize"]
                main(options)
                subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()

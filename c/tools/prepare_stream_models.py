#!/usr/bin/env python3
"""Explicitly prepare hash-verified compact models; never run at API startup."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SPEECH_REPO = "OpenVINO/whisper-tiny-int8-ov"
SPEECH_REVISION = "a850762d97243dee30f46ca309720541af619ab0"
TEXT_REPO = "OpenVINO/Qwen2-0.5B-Instruct-int4-ov"
TEXT_REVISION = "e1152fa9f77f3fa6a3f19b4210529a65e549e72b"
CAMERA_SOURCE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/single-image-super-resolution-1032/FP16/"
MIB = 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(kind, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a new directory; existing models are preserved")
    output.mkdir(parents=True)
    try:
        if kind in ("speech", "text_graph"):
            from huggingface_hub import snapshot_download
            # Public checkpoint, explicit revision, no remote Python or auth token.
            repo, revision = (SPEECH_REPO, SPEECH_REVISION) if kind == "speech" else (TEXT_REPO, TEXT_REVISION)
            snapshot_download(repo, revision=revision, token=False,
                              local_dir=output, cache_dir=output.parent / ".download-cache",
                              allow_patterns=["*.json", "*.xml", "*.bin", "*.txt"])
            family, adapter, precision = "speech", "whisper_openvino", "int8"
            options = {"max_audio_seconds": 30, "model_path": ".", "max_output_bytes": MIB}
            resident, workspace, session = 256 * MIB, 128 * MIB, 8 * MIB
            provenance = {"source": SPEECH_REPO, "revision": SPEECH_REVISION,
                          "license": "Apache-2.0 (checkpoint card)", "task": "recognition/speech-to-English"}
            max_input = 2 * MIB
            if kind == "text_graph":
                family, adapter, precision = "text_graph", "text_openvino", "int4"
                options = {"max_tokens": 256, "max_prompt_tokens": 2048,
                           "max_context_tokens": 4096, "max_output_bytes": MIB}
                resident, workspace, session = 640 * MIB, 256 * MIB, 8 * MIB
                provenance = {"source": repo, "revision": revision,
                              "license": "Apache-2.0 (checkpoint card)", "task": "text generation"}
                max_input = 64 * 1024
        elif kind == "camera_upscale":
            stem = "single-image-super-resolution-1032"
            for suffix in ("xml", "bin"):
                request = urllib.request.Request(CAMERA_SOURCE + stem + "." + suffix,
                                                 headers={"User-Agent": "colibri-stream-model-preparation"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read(4 * MIB + 1)
                if not data or len(data) > 4 * MIB:
                    raise ValueError("camera artifact is empty or exceeds preparation limit")
                (output / ("model." + suffix)).write_bytes(data)
            family, adapter, precision = "camera_upscale", "super_resolution_openvino", "fp16"
            options = {"model_xml": "model.xml", "max_output_pixels": 5592405,
                       "scale": 4, "max_output_bytes": 32 * MIB}
            resident, workspace, session = 64 * MIB, 128 * MIB, MIB
            provenance = {"source": CAMERA_SOURCE, "revision": "open_model_zoo/2023.0/models_bin/1",
                          "license": "Open Model Zoo model terms", "task": "trained 4x super-resolution"}
            max_input = 4 * MIB
        else:
            raise ValueError("unknown runtime model kind")
        files = {str(path.relative_to(output)).replace("\\", "/"): digest(path)
                 for path in output.rglob("*") if path.is_file() and ".cache" not in path.parts}
        if not files or sum((output / name).stat().st_size for name in files) > 512 * MIB:
            raise ValueError("model asset inventory empty or exceeds 512 MiB preparation limit")
        manifest = {"schema": "colibri.runtime-model/v1", "family": family,
                    "model_id": "colibri-" + family, "adapter": adapter,
                    "files": files, "precision": precision, "max_input_bytes": max_input,
                    "resident_bytes": resident, "workspace_bytes": workspace,
                    "session_bytes": session, "options": options, "provenance": provenance}
        (output / "colibri_runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except BaseException:
        # Keep partial downloads for inspection/resume; never erase user artifacts.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("speech", "camera_upscale", "text_graph"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = prepare(args.kind, args.output)
        print(json.dumps({"directory": str(args.output), "family": manifest["family"], "files": len(manifest["files"])}))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"model preparation failed: {error}\n")


if __name__ == "__main__":
    main()

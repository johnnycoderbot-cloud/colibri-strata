"""Explicit file/replay and opt-in capture commands for Colibri graph models."""
import base64
from fractions import Fraction
import io
import json
from pathlib import Path
import stat
import time


def budget(args):
    if getattr(args, "attach", None):
        raise ValueError("graph chat --attach is unsupported; use its HTTP chat endpoint with stream:false or local chat")
    native_flags = ("auto_tier", "no_tune_profile", "ctx", "gpu", "vram", "repin", "xdna", "cap",
                    "topk", "topp", "cluster_workers", "cluster_coordinator")
    for name in native_flags:
        if getattr(args, name, None):
            raise ValueError(f"--{name.replace('_', '-')} applies to native engines; graph packs use --device and --memory-budget-mib")
    for name in ("temp", "think", "effort"):
        if getattr(args, name, None) is not None:
            raise ValueError(f"--{name} is unsupported by these compact deterministic graph adapters")
    if getattr(args, "policy", "quality") != "quality":
        raise ValueError("graph packs use manifest admission estimates; native --policy is unsupported")
    value = getattr(args, "memory_budget_mib", 1024)
    ram = getattr(args, "ram", 0)
    if ram:
        if type(ram) is not int or ram < 1 or value != 1024:
            raise ValueError("select one positive --ram GiB or --memory-budget-mib budget")
        value = ram * 1024
    if type(value) is not int or not 1 <= value <= 65536:
        raise ValueError("--memory-budget-mib must be in [1, 65536]")
    return value * 1024 * 1024


def _engine(args):
    from stream_runtime import GraphEngine
    return GraphEngine(args.model, device=getattr(args, "device", "AUTO"), memory_budget_bytes=budget(args))


def inspect_model(args, compile_model=False):
    from stream_runtime import validate_manifest
    manifest, root, assets, estimated = validate_manifest(args.model, budget(args))
    result = {"runtime_kind": "graph", "family": manifest["family"], "model_id": manifest["model_id"],
              "asset_bytes": assets, "base_estimated_bytes": estimated, "memory_budget_bytes": budget(args),
              "memory_estimates_only": True, "files_verified": len(manifest["files"]),
              "requested_device": args.device, "provenance": manifest.get("provenance", {})}
    if compile_model:
        engine = _engine(args)
        try:
            result["backend"] = engine.info()
        finally:
            engine.close()
    print(json.dumps(result, indent=2))


def _file(path, maximum=4 << 20):
    from stream_runtime import _read_checked
    # Regular-file/no-symlink bounded reader also rejects FIFOs before opening.
    target = Path(path)
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
        raise ValueError("input must be a nonempty bounded regular file")
    return _read_checked(target, metadata, maximum)


def _png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    data = stream.getvalue()
    if len(data) > 4 << 20:
        raise ValueError("encoded input frame exceeds 4 MiB")
    return data


def _image(path):
    from PIL import Image
    raw = _file(path)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in ("PNG", "JPEG", "WEBP") or image.width * image.height > 349525:
            raise ValueError("image must be PNG/JPEG/WEBP with at most 349525 source pixels")
        image.load()
        return _png_bytes(image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"))


def _output(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)


def run_text(args, interactive=False):
    engine = _engine(args)
    try:
        if engine.modality != "text":
            raise ValueError("use transcribe for speech or upscale for images/video/camera")
        tokens = getattr(args, "ngen", None) or 128
        if not interactive:
            prompt = " ".join(args.prompt)
            print(engine.complete(prompt, max_new_tokens=tokens)["text"])
            return
        messages = []
        while True:
            try:
                prompt = input("you> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip() in ("/quit", "/exit"):
                break
            if not prompt.strip():
                continue
            messages.append({"role": "user", "content": prompt})
            try:
                result = engine.chat(messages, max_new_tokens=tokens)
            except (ValueError, RuntimeError):
                messages.pop()
                raise
            print(result["text"])
            messages.append({"role": "assistant", "content": result["text"]})
    finally:
        engine.close()


def transcribe(args):
    engine = _engine(args)
    try:
        if getattr(args, "microphone", False):
            if args.input:
                raise ValueError("select a WAV file or --microphone, not both")
            import sounddevice as sd
            import numpy as np
            import wave
            if type(args.seconds) not in (float, int) or not 0 < args.seconds < 30:
                raise ValueError("microphone capture seconds must be in (0, 30)")
            samples = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1, dtype="int16")
            sd.wait()
            stream = io.BytesIO()
            with wave.open(stream, "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
            data = stream.getvalue()
        else:
            if not args.input:
                raise ValueError("provide a WAV file or explicitly select --microphone")
            data = _file(args.input, 2 << 20)
        result = engine.transcribe(data, task="translate" if args.translate else "transcribe",
                                   language=None if args.auto_language or args.translate else "en")
        print(json.dumps({**result, "runtime": engine.info()}, ensure_ascii=False))
    finally:
        engine.close()


def _video(engine, source, output):
    import av
    import numpy as np
    from PIL import Image
    if not stat.S_ISREG(Path(source).lstat().st_mode):
        raise ValueError("video input must be a local regular file")
    # Exclusive creation prevents replacing a user's original media.
    target_file = Path(output).open("xb")
    frames = 0
    try:
        with target_file, av.open(str(source)) as reader, av.open(target_file, mode="w",
                format="mp4" if str(output).lower().endswith(".mp4") else "matroska") as writer:
            videos = list(reader.streams.video)
            if len(videos) != 1:
                raise ValueError("video input must contain exactly one video stream")
            original = videos[0]
            if original.width * original.height > 349525:
                raise ValueError("video source frames exceed compact model pixel limit")
            enhanced = writer.add_stream("libx264", rate=original.average_rate or 30)
            enhanced.width, enhanced.height = original.width * 4, original.height * 4
            enhanced.pix_fmt = "yuv420p"
            tick = Fraction(1, 90000)
            enhanced.time_base = enhanced.codec_context.time_base = tick
            audio = {s.index: writer.add_stream_from_template(s) for s in reader.streams.audio}
            for packet in reader.demux([original, *reader.streams.audio]):
                if packet.stream.type == "audio":
                    if packet.dts is not None:
                        packet.stream = audio[packet.stream.index]
                        writer.mux(packet)
                    continue
                for frame in packet.decode():
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError("video frames need source timestamps")
                    with frame.to_image() as picture:
                        result = engine.upscale(_png_bytes(picture))
                    with Image.open(io.BytesIO(base64.b64decode(result["png_base64"], validate=True))) as picture:
                        converted = av.VideoFrame.from_ndarray(np.asarray(picture.convert("RGB")), format="rgb24")
                    converted.pts = round(frame.pts * frame.time_base / tick)
                    converted.time_base = tick
                    for encoded in enhanced.encode(converted):
                        writer.mux(encoded)
                    frames += 1
            for encoded in enhanced.encode():
                writer.mux(encoded)
        if not frames:
            raise ValueError("video contains no decoded frames")
        return {"frames": frames, "audio_streams": len(audio), "source_timestamps_preserved": True}
    except BaseException:
        # Retain partial output for inspection; never claim an incomplete file succeeded.
        target_file.close()
        raise


def _camera(engine, args):
    import cv2
    from PIL import Image
    if not 1 <= args.frames <= 10000:
        raise ValueError("camera --frames must be in [1,10000]")
    directory = Path(args.output)
    directory.mkdir(exist_ok=False, parents=True)
    capture = cv2.VideoCapture(args.camera)
    session = None
    try:
        if not capture.isOpened():
            raise RuntimeError("camera could not be opened")
        session = engine.create_stream("camera", queue_capacity=2)
        origin = time.monotonic()
        finished = False
        for sequence in range(args.frames):
            ok, pixels = capture.read()
            if not ok:
                raise RuntimeError("camera frame read failed")
            with Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)) as image:
                payload = _png_bytes(image)
            timestamp = (time.monotonic() - origin) * 1000
            while True:
                try:
                    engine.push_stream(session, sequence, timestamp, payload, final=sequence == args.frames - 1)
                    break
                except BufferError:
                    for event in engine.stream_events(session, max_events=4, wait_ms=1000):
                        _camera_event(directory, event)
            for event in engine.stream_events(session, max_events=4):
                _camera_event(directory, event)
                finished |= event["type"] in ("finished", "error", "cancelled")
        while not finished:
            events = engine.stream_events(session, max_events=4, wait_ms=1000)
            for event in events:
                _camera_event(directory, event)
            if any(event["type"] in ("closed", "finished", "error", "cancelled") for event in events):
                finished = True
    finally:
        capture.release()
        if session:
            engine.close_stream(session)


def _camera_event(directory, event):
    if event.get("type") == "result":
        result = event["result"]
        _output(directory / f"frame-{event['sequence']:06d}.png", base64.b64decode(result["png_base64"], validate=True))
        event = {**event, "result": {key: value for key, value in result.items() if key != "png_base64"}}
    if event.get("type") == "error":
        raise RuntimeError(event.get("error", "camera inference failed"))
    print(json.dumps(event))


def upscale(args):
    engine = _engine(args)
    try:
        if engine.modality != "camera":
            raise ValueError("upscale requires a camera_upscale model pack")
        if args.camera is not None:
            if args.input or args.video or not 0 <= args.camera <= 31:
                raise ValueError("select only --camera INDEX in [0,31] for live capture")
            return _camera(engine, args)
        if not args.input:
            raise ValueError("provide an image/video file or explicitly select --camera INDEX")
        if args.video:
            result = _video(engine, args.input, args.output)
        else:
            result = engine.upscale(_image(args.input))
            _output(args.output, base64.b64decode(result.pop("png_base64"), validate=True))
        print(json.dumps({**result, "output": args.output, "runtime": engine.info()}))
    finally:
        engine.close()


def prepare(args):
    from tools.prepare_stream_models import prepare as prepare_model
    result = prepare_model(args.kind, args.output)
    print(json.dumps({"family": result["family"], "directory": args.output}))

#!/usr/bin/env python3
"""Replay an installed model through the actual authenticated gateway; no capture."""
import argparse
import base64
import http.client
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server
from stream_runtime import GraphEngine


def check(model, device, replay_input=None):
    with GraphEngine(model, device=device) as engine:
        openai_server.ARCH = engine.info()["family"]
        server = openai_server.APIServer(("127.0.0.1", 0), engine, engine.model_id, "local-replay-key", 64)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def request(path, body=None, auth=True):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
            try:
                headers = {"Content-Type": "application/json"}
                if auth:
                    headers["Authorization"] = "Bearer local-replay-key"
                connection.request("POST" if body else "GET", path, json.dumps(body) if body else None, headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read(36 << 20))
            finally:
                connection.close()
        try:
            assert request("/v1/models", auth=False)[0] == 401
            status, card = request("/v1/models")
            assert status == 200 and card["data"][0]["runtime"]["family"] == engine.info()["family"]
            body = {"model": engine.model_id}
            if engine.modality == "text":
                path = "/v1/chat/completions"
                body.update(messages=[{"role": "user", "content": "Name the capital of France. Answer with the city only."}], max_tokens=24)
            else:
                if replay_input is None:
                    raise ValueError("audio/image replay needs --input")
                data = Path(replay_input).read_bytes()
                if engine.modality == "audio":
                    path = "/v1/audio/transcriptions"
                    body["audio_base64"] = base64.b64encode(data).decode("ascii")
                else:
                    path = "/v1/camera/upscale"
                    body["png_base64"] = base64.b64encode(data).decode("ascii")
            started = time.perf_counter()
            status, result = request(path, body)
            elapsed = time.perf_counter() - started
            if status != 200:
                raise RuntimeError(f"actual gateway replay failed: {status} {result}")
            if engine.modality == "text" and "Paris" not in result["choices"][0]["message"]["content"]:
                raise RuntimeError("text replay correctness check failed")
            if engine.modality == "audio" and "ask not" not in result.get("text", "").lower():
                raise RuntimeError("JFK audio replay correctness check failed")
            if engine.modality == "camera" and not (result.get("scale") == 4 and result.get("png_base64")):
                raise RuntimeError("camera replay correctness check failed")
            result.pop("png_base64", None)
            return {"schema": "colibri.actual-gateway-replay/v1", "model": engine.model_id,
                    "device": device, "endpoint": path, "http_status": status,
                    "elapsed_seconds": elapsed, "auth_refusal_passed": True,
                    "runtime": engine.info(), "result": result}
        finally:
            server.shutdown()
            server.scheduler.close()
            server.server_close()
            worker.join(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--input")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        parser.error("report must be a new file")
    report = check(args.model, args.device, args.input)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"model": report["model"], "http_status": report["http_status"], "device": args.device}))


if __name__ == "__main__":
    main()

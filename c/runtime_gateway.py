"""Graph workload routes for the existing authenticated Colibri gateway."""
import base64
import binascii
from email import policy
from email.parser import BytesParser
import json
import time
import uuid
from urllib.parse import parse_qs, urlsplit


def is_graph_engine(engine):
    return getattr(engine, "runtime_kind", None) == "graph"


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _finite(value):
    raise ValueError("nonfinite JSON number")


def decode_payload(value, limit):
    if not isinstance(value, str) or len(value) > 4 * ((limit + 2) // 3):
        raise ValueError("payload must be bounded base64")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid base64 payload") from error
    if not data or len(data) > limit:
        raise ValueError("payload is empty or exceeds the model input bound")
    return data


class RuntimeRoutes:
    def runtime_error(self, status, message):
        # Handler owns the error class, including when server runs as __main__.
        return self.api_error(status, message)

    def read_runtime_body(self):
        raw = self.read_raw_body()
        content_type = self.headers.get("Content-Type", "application/json")
        if content_type.split(";", 1)[0].strip().lower() == "multipart/form-data":
            if len(content_type) > 512 or "\r" in content_type or "\n" in content_type:
                raise self.runtime_error(400, "Invalid multipart Content-Type.")
            if not content_type.isascii():
                raise self.runtime_error(400, "Multipart Content-Type must be ASCII.")
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode("ascii") + raw)
            if (not message.is_multipart() or message.defects or
                    any(getattr(header, "defects", ()) for _, header in message.items())):
                raise self.runtime_error(400, "Invalid multipart body or boundary.")
            body = {}
            for index, part in enumerate(message.iter_parts()):
                name = part.get_param("name", header="content-disposition")
                if (index >= 8 or part.is_multipart() or part.defects or
                        any(getattr(header, "defects", ()) for _, header in part.items()) or
                        part.get_content_disposition() != "form-data" or
                        name not in ("file", "model", "language", "response_format") or name in body or
                        (name == "file" and "audio_base64" in body) or
                        len(part.get_all("Content-Disposition", [])) != 1 or
                        part.get("Content-Transfer-Encoding") is not None):
                    raise self.runtime_error(400, "Unsupported or duplicate multipart field.")
                value = part.get_payload(decode=True)
                if name == "file":
                    body["audio_base64"] = base64.b64encode(value).decode("ascii")
                else:
                    if len(value) > 1024:
                        raise self.runtime_error(400, "Multipart text field is too large.")
                    try:
                        body[name] = value.decode("utf-8", "strict")
                    except UnicodeError as error:
                        raise self.runtime_error(400, "Multipart fields must be UTF-8.") from error
            if "audio_base64" not in body:
                raise self.runtime_error(400, "Multipart audio requires a file field.")
            return body
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise self.runtime_error(415, "Use application/json or multipart/form-data.")
        try:
            body = json.loads(raw, object_pairs_hook=_unique, parse_constant=_finite)
            json.dumps(body, allow_nan=False, ensure_ascii=False).encode("utf-8")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            return body
        except (ValueError, UnicodeError, RecursionError) as error:
            raise self.runtime_error(400, "Request must contain a finite, unique-field UTF-8 JSON object.") from error

    def runtime_get(self, path, request_id):
        pieces = path.split("/")
        if len(pieces) != 6 or pieces[1:4] != ["v1", "runtime", "streams"] or pieces[5] != "events":
            return False
        try:
            query = parse_qs(urlsplit(self.path).query, strict_parsing=True, keep_blank_values=True)
            if set(query) - {"max_events", "wait_ms"} or any(len(v) != 1 for v in query.values()):
                raise ValueError("invalid events query")
            maximum = int(query.get("max_events", ["32"])[0])
            wait = int(query.get("wait_ms", ["0"])[0])
            if not 1 <= maximum <= 64 or not 0 <= wait <= 1000:
                raise ValueError("event bounds exceeded")
            events = self.server.engine.stream_events(pieces[4], maximum, wait)
        except KeyError as error:
            raise self.runtime_error(404, "Unknown stream.") from error
        except (TypeError, ValueError, RuntimeError) as error:
            raise self.runtime_error(400, str(error)) from error
        self.send_json(200, {"events": events}, request_id)
        return True

    def runtime_post(self, path, body, request_id):
        engine = self.server.engine
        try:
            if path in ("/v1/completions", "/v1/chat/completions"):
                if engine.modality != "text":
                    raise ValueError("this runtime model does not generate text")
                chat = path == "/v1/chat/completions"
                supported = {"model", "messages" if chat else "prompt", "max_tokens", "stream"}
                if set(body) - supported or body.get("stream", False) is not False:
                    raise ValueError("compact text accepts model, prompt/messages and max_tokens; stream must be false")
                count = body.get("max_tokens", min(self.server.max_tokens, 128))
                if type(count) is not int or not 1 <= count <= min(self.server.max_tokens, 256):
                    raise ValueError("max_tokens exceeds the compact text limit")
                with self.server.scheduler.admit(self.client_disconnected):
                    result = (engine.chat(body.get("messages"), max_new_tokens=count, cancel=self.client_disconnected)
                              if chat else engine.complete(body.get("prompt"), max_new_tokens=count, cancel=self.client_disconnected))
                # GenAI's string result does not expose an EOS-vs-limit reason.
                choice = {"index": 0, "finish_reason": None}
                choice["message" if chat else "text"] = ({"role": "assistant", "content": result["text"]} if chat else result["text"])
                self.send_json(200, {"id": "chatcmpl_" + uuid.uuid4().hex,
                                    "object": "chat.completion" if chat else "text_completion",
                                    "created": int(time.time()), "model": self.server.model_id,
                                    "choices": [choice], "runtime": result}, request_id)
            elif path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
                if engine.modality != "audio":
                    raise ValueError("this runtime model does not transcribe audio")
                if body.get("response_format", "json") != "json":
                    raise ValueError("response_format must be json")
                if set(body) - {"model", "audio_base64", "language", "response_format"}:
                    raise ValueError("unsupported audio field")
                payload = decode_payload(body.get("audio_base64"), engine.info()["max_input_bytes"])
                with self.server.scheduler.admit(self.client_disconnected):
                    result = engine.transcribe(payload, task="translate" if path.endswith("translations") else "transcribe",
                                               language=body.get("language", None if path.endswith("translations") else "en"),
                                               cancel=self.client_disconnected)
                self.send_json(200, result, request_id)
            elif path == "/v1/camera/upscale":
                if engine.modality != "camera":
                    raise ValueError("this runtime model does not upscale images")
                if set(body) - {"model", "png_base64"}:
                    raise ValueError("unsupported upscale field")
                payload = decode_payload(body.get("png_base64"), engine.info()["max_input_bytes"])
                with self.server.scheduler.admit(self.client_disconnected):
                    result = engine.upscale(payload, cancel=self.client_disconnected)
                self.send_json(200, result, request_id)
            elif path == "/v1/runtime/streams":
                if set(body) - {"model", "kind", "queue_capacity", "options"}:
                    raise ValueError("unsupported stream field")
                stream_id = engine.create_stream(body.get("kind"), body.get("queue_capacity", 2), body.get("options", {}))
                self.send_json(201, {"stream_id": stream_id}, request_id)
            elif path.startswith("/v1/runtime/streams/"):
                parts = path.split("/")
                if len(parts) != 6:
                    raise self.runtime_error(404, "Unknown runtime endpoint.")
                stream_id, action = parts[4:]
                if action == "push":
                    if set(body) - {"model", "sequence", "timestamp_ms", "payload_base64", "final"}:
                        raise ValueError("unsupported frame field")
                    payload = decode_payload(body.get("payload_base64"), engine.info()["max_input_bytes"])
                    result = engine.push_stream(stream_id, body.get("sequence"), body.get("timestamp_ms"), payload,
                                                final=body.get("final", False))
                elif action == "cancel":
                    if set(body) - {"model"}:
                        raise ValueError("unsupported cancellation field")
                    engine.cancel_stream(stream_id)
                    result = {"cancelled": True}
                elif action == "close":
                    if set(body) - {"model"}:
                        raise ValueError("unsupported close field")
                    result = {"closed": engine.close_stream(stream_id, timeout=1)}
                else:
                    raise self.runtime_error(404, "Unknown runtime endpoint.")
                self.send_json(200, result, request_id)
            else:
                raise self.runtime_error(400, "Unsupported endpoint for this graph model; inspect /v1/models capabilities.")
        except KeyError as error:
            raise self.runtime_error(404, "Unknown stream.") from error
        except (BufferError, MemoryError) as error:
            raise self.runtime_error(429, str(error)) from error
        except (TypeError, ValueError) as error:
            raise self.runtime_error(400, str(error)) from error
        except RuntimeError as error:
            from stream_runtime import GraphCancelled
            from stream_models import StreamCancelled
            if isinstance(error, (GraphCancelled, StreamCancelled)):
                raise self.runtime_error(409, "Operation cancelled.") from error
            raise self.runtime_error(503, "Graph backend could not complete the operation.") from error

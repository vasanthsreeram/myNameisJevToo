"""Local ``POST /v1/systemone``.

One loaded model, the published request shape, and a lock around the forward.
MLX keeps a single Metal queue, so concurrent requests wait instead of
interleaving graphs. The wait shows up as ``queue_ms``.

    python -m jevtoo.serve --model openbmb/MiniCPM5-2B-MLX
    python -m jevtoo.serve --gguf http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac

from .contract import ContractError, evaluate
from .decide import DecisionModel

MAX_BODY = 8_000_000
DEFAULT_ALIASES = frozenset({"jev-latest", "jev-preview"})


@dataclass
class App:
    decision: DecisionModel
    model_id: str
    backend_name: str
    api_key: str | None = None
    aliases: frozenset[str] = DEFAULT_ALIASES
    infer_lock: threading.Lock = field(default_factory=threading.Lock)
    metrics_lock: threading.Lock = field(default_factory=threading.Lock)
    metrics: dict = field(default_factory=lambda: {
        "requests": 0, "errors": 0, "questions": 0,
        "queue_ms": 0.0, "service_ms": 0.0, "forward_ms": 0.0,
    })


def release_allocator(backend: object) -> None:
    """Drop the logits buffer MLX would otherwise keep for the process lifetime."""
    mx = getattr(backend, "mx", None)
    clear = getattr(mx, "clear_cache", None) if mx is not None else None
    if callable(clear):
        clear()


def _authorized(app: App, header: str | None) -> bool:
    if not app.api_key:
        return True
    if not header or not header.startswith("Bearer "):
        return False
    presented = header[len("Bearer "):].strip()
    return hmac.compare_digest(presented, app.api_key)


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict, extra: dict | None = None) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.send_header("Access-Control-Allow-Origin", "*")
    for key, value in (extra or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def _note(app: App, *, error: bool, questions: int, queue_ms: float, service_ms: float, forward_ms: float) -> None:
    with app.metrics_lock:
        app.metrics["requests"] += 1
        app.metrics["errors"] += int(error)
        app.metrics["questions"] += questions
        app.metrics["queue_ms"] += queue_ms
        app.metrics["service_ms"] += service_ms
        app.metrics["forward_ms"] += forward_ms


def _log(record: dict) -> None:
    sys.stderr.write(json.dumps(record, default=str) + "\n")
    sys.stderr.flush()


def build_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Jev-Observe")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/v1/health"):
                stats = getattr(app.decision.backend, "stats", {}) or {}
                cov = getattr(app.decision.backend, "coverage", None)
                _json(self, 200, {
                    "status": "ok",
                    "model": app.model_id,
                    "backend": app.backend_name,
                    "endpoint": "/v1/systemone",
                    "aliases": sorted(app.aliases),
                    "head_mode": getattr(app.decision.backend, "_head_mode", None) or stats.get("head_mode"),
                    "decisions": stats.get("decisions", 0),
                    "coverage": cov() if callable(cov) else None,
                    "auth": "bearer" if app.api_key else "open",
                })
                return
            if path == "/v1/models":
                _json(self, 200, {"models": [
                    {
                        "id": app.model_id,
                        "description": f"Open model loaded in this process ({app.backend_name}).",
                    },
                    *[
                        {"id": alias, "description": f"Alias of {app.model_id}.", "points_to": app.model_id}
                        for alias in sorted(app.aliases)
                    ],
                ]})
                return
            if path == "/metrics":
                if not _authorized(app, self.headers.get("Authorization")):
                    _json(self, 401, {"error": {
                        "message": "Missing or invalid API key.",
                        "type": "authentication_error",
                    }})
                    return
                with app.metrics_lock:
                    snap = dict(app.metrics)
                lines = [
                    "# TYPE jevtoo_requests_total counter",
                    f"jevtoo_requests_total {snap['requests']}",
                    "# TYPE jevtoo_errors_total counter",
                    f"jevtoo_errors_total {snap['errors']}",
                    "# TYPE jevtoo_questions_total counter",
                    f"jevtoo_questions_total {snap['questions']}",
                    "# TYPE jevtoo_queue_ms_sum counter",
                    f"jevtoo_queue_ms_sum {snap['queue_ms']:.3f}",
                    "# TYPE jevtoo_service_ms_sum counter",
                    f"jevtoo_service_ms_sum {snap['service_ms']:.3f}",
                    "# TYPE jevtoo_forward_ms_sum counter",
                    f"jevtoo_forward_ms_sum {snap['forward_ms']:.3f}",
                    "",
                ]
                body = "\n".join(lines).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Connection", "close")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/":
                _json(self, 200, {
                    "service": "myNameisJevToo",
                    "endpoint": "POST /v1/systemone",
                    "model": app.model_id,
                    "health": "/health",
                    "metrics": "/metrics",
                })
                return
            _json(self, 404, {"error": {"message": f"no route {path}", "type": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/v1/systemone":
                _json(self, 404, {"error": {"message": f"no route {path}", "type": "not_found"}})
                return
            request_id = uuid.uuid4().hex
            if not _authorized(app, self.headers.get("Authorization")):
                _note(app, error=True, questions=0, queue_ms=0, service_ms=0, forward_ms=0)
                _json(self, 401, {"error": {
                    "message": "Missing or invalid API key. Send Authorization: Bearer <JEVTOO_API_KEY>.",
                    "type": "authentication_error",
                }}, {"X-Request-Id": request_id})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = 0
            if length > MAX_BODY:
                _json(self, 422, {"error": {
                    "message": f"body exceeds {MAX_BODY} bytes",
                    "type": "invalid_request_error",
                }}, {"X-Request-Id": request_id})
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode() or "null")
            except (UnicodeDecodeError, json.JSONDecodeError):
                _note(app, error=True, questions=0, queue_ms=0, service_ms=0, forward_ms=0)
                _json(self, 422, {"error": {
                    "message": "request body is not JSON",
                    "type": "invalid_request_error",
                }}, {"X-Request-Id": request_id})
                return

            observe = (self.headers.get("X-Jev-Observe") or "").strip().lower() in ("1", "true", "yes")
            if "observe=1" in self.path or "observe=true" in self.path:
                observe = True

            t_wait = time.perf_counter()
            with app.infer_lock:
                queue_s = time.perf_counter() - t_wait
                t0 = time.perf_counter()
                try:
                    payload, obs = evaluate(
                        app.decision, body,
                        model_id=app.model_id, aliases=set(app.aliases), observe=observe,
                    )
                    status = 200
                    error = False
                except ContractError as exc:
                    payload = {"error": {
                        "message": exc.message,
                        "type": "invalid_request_error",
                        **({"field": exc.field} if exc.field else {}),
                    }}
                    obs = {"questions": {}}
                    status = exc.status
                    error = True
                except ValueError as exc:
                    payload = {"error": {"message": str(exc), "type": "invalid_request_error"}}
                    obs = {"questions": {}}
                    status = 422
                    error = True
                except Exception as exc:  # noqa: BLE001
                    payload = {"error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "type": "server_error",
                    }}
                    obs = {"questions": {}}
                    status = 500
                    error = True
                    _log({"event": "error", "request_id": request_id, "error": repr(exc)})
                finally:
                    release_allocator(app.decision.backend)
                service_s = time.perf_counter() - t0

            obs["request_id"] = request_id
            obs["queue_ms"] = round(queue_s * 1000, 3)
            obs["service_ms"] = round(service_s * 1000, 3)
            if observe and "observability" in payload:
                payload["observability"]["request_id"] = request_id
                payload["observability"]["queue_ms"] = obs["queue_ms"]
                payload["observability"]["service_ms"] = obs["service_ms"]
            forward_ms = float(obs.get("forward_ms") or 0.0)
            _note(
                app, error=error, questions=len(obs.get("questions") or {}),
                queue_ms=obs["queue_ms"], service_ms=obs["service_ms"], forward_ms=forward_ms,
            )
            _log({
                "event": "systemone",
                "request_id": request_id,
                "status": status,
                "model": app.model_id,
                "questions": sorted((obs.get("questions") or {}).keys()),
                "queue_ms": obs["queue_ms"],
                "service_ms": obs["service_ms"],
                "forward_ms": forward_ms,
                "shared_prefix_tokens": obs.get("shared_prefix_tokens", 0),
            })
            timing = (
                f"queue;dur={obs['queue_ms']:.3f}, "
                f"service;dur={obs['service_ms']:.3f}, "
                f"forward;dur={forward_ms:.3f}"
            )
            _json(self, status, payload, {
                "X-Request-Id": request_id,
                "Server-Timing": timing,
            })

    return Handler


def create_server(decision: DecisionModel, model_id: str, host: str = "127.0.0.1", port: int = 8787,
                  api_key: str | None = None, backend_name: str = "mlx",
                  aliases: frozenset[str] | None = None) -> ThreadingHTTPServer:
    app = App(
        decision=decision, model_id=model_id, backend_name=backend_name,
        api_key=api_key, aliases=aliases if aliases is not None else DEFAULT_ALIASES,
    )
    httpd = ThreadingHTTPServer((host, port), build_handler(app))
    httpd.app = app  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve any causal LM at POST /v1/systemone")
    parser.add_argument("--model", help="Hugging Face id or local MLX model path")
    parser.add_argument("--gguf", help="Base URL of a running llama-server, e.g. http://127.0.0.1:8080")
    parser.add_argument("--model-id", help="Id returned in responses. Defaults to --model or --gguf.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--api-key", default=os.environ.get("JEVTOO_API_KEY"),
                        help="If set, require Authorization: Bearer. Defaults to $JEVTOO_API_KEY.")
    parser.add_argument("--n-probs", type=int, default=200, help="GGUF top-N window for label probabilities")
    args = parser.parse_args(argv)
    if bool(args.model) == bool(args.gguf):
        parser.error("pass exactly one of --model or --gguf")

    if args.gguf:
        from .decide import convert_gguf
        decision = convert_gguf(args.gguf, n_probs=args.n_probs)
        model_id = args.model_id or args.gguf
        backend_name = "gguf"
    else:
        from .decide import convert
        decision = convert(args.model)
        model_id = args.model_id or args.model
        backend_name = "mlx"

    httpd = create_server(
        decision, model_id, host=args.host, port=args.port,
        api_key=args.api_key or None, backend_name=backend_name,
    )
    bound = httpd.server_address
    print(f"jevtoo  POST http://{bound[0]}:{bound[1]}/v1/systemone  model={model_id}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()

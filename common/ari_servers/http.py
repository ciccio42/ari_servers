"""HTTP service shared by the servers: routes, JSON replies, errors, keep-alive, TCP_NODELAY. Standard library only.

    service = HTTPService("whisper", host, port, execution)
    service.add_route("POST", "/transcribe", handler)   # handler(request) -> dict (200 JSON) or None (replied)
    service.serve_forever()

GET /health is always there: {"status": "ok", "service": ..., **service.health()}.
A handler raises HTTPError(code, message) for the errors of the client; any other exception becomes a 500 and the
server keeps running.
"""
import json
import logging
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class HTTPError(Exception):

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class Request(BaseHTTPRequestHandler):
    server_version = "AriServer/1.0"
    protocol_version = "HTTP/1.1"  # keep-alive: the clients reuse the connection

    def setup(self):
        super().setup()
        # headers and body of the reply go out in two writes: with Nagle the second waits for the delayed ACK of the
        # client, about 40 ms per request (measured on the Whisper server)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt, *args):
        self.server.service.log.debug("%s %s", self.address_string(), fmt % args)

    # --- for the handlers
    @property
    def route_path(self) -> str:
        return urlsplit(self.path).path

    def read_body(self, max_bytes: int) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > max_bytes:
            raise HTTPError(400, f"Content-Length must be between 1 and {max_bytes}")
        return self.rfile.read(length)

    def read_json(self, max_bytes: int = 1024 * 1024) -> dict:
        try:
            data = json.loads(self.read_body(max_bytes))
        except ValueError as e:
            raise HTTPError(400, f"invalid JSON: {e}")
        if not isinstance(data, dict):
            raise HTTPError(400, "the body must be a JSON object")
        return data

    def reply_json(self, code: int, payload: dict):
        self.reply_raw(code, json.dumps(payload).encode(), "application/json")

    def reply_raw(self, code: int, body: bytes, content_type: str, headers: dict = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    # --- dispatch
    def _dispatch(self, method: str):
        self.t_in = time.perf_counter()
        service = self.server.service
        handler = service.routes.get((method, self.route_path)) or service.fallback
        if handler is None:
            self.reply_json(404, {"error": f"unknown path {method} {self.route_path}"})
            return
        try:
            result = handler(self)
        except HTTPError as e:
            self.reply_json(e.code, {"error": e.message})
            return
        except NotImplementedError as e:
            self.reply_json(501, {"error": f"not implemented yet: {e}"})
            return
        except Exception as e:  # a failed request must not stop the server
            service.log.exception("%s %s failed", method, self.route_path)
            self.reply_json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        if result is not None:
            self.reply_json(200, result)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")


class HTTPService:

    def __init__(self, name: str, host: str, port: int, execution=None):
        self.name, self.host, self.port, self.execution = name, host, port, execution
        self.log = logging.getLogger(name)
        self.routes = {("GET", "/health"): lambda request: {"status": "ok", "service": self.name, **self.health()}}
        self.fallback = None  # handler of every other path (e.g. a proxy)

    def add_route(self, method: str, path: str, handler):
        self.routes[(method, path)] = handler

    def health(self) -> dict:
        """Extra fields of GET /health (model, language, ...)."""
        return {}

    def serve_forever(self):
        server = ThreadingHTTPServer((self.host, self.port), Request)
        server.daemon_threads = True
        server.service = self
        lock = self.execution.describe() if self.execution else "none"
        self.log.info("%s listening on http://%s:%d (exclusive execution: %s)", self.name, self.host, self.port, lock)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")

"""RASA server of the architecture: a proxy in front of Rasa, so Rasa follows "no concurrent executions" too.

Rasa has its own HTTP API (rasa run --enable-api): the clients of the demo keep using it unchanged
(/webhooks/rest/webhook, /conversations/<id>/tracker/events, /model/parse, /), on the port of this proxy.
Every request is forwarded as it is to RasaProxyConfig.upstream, holding the execution lock of the servers.

  GET /health   {"status": "ok", "service": "rasa_proxy", "upstream": ..., "upstream_ok": ...}
  any other     forwarded to Rasa; 502 if Rasa does not answer

Run: python -m rasa_proxy [--upstream ...] [--port ...] (PYTHONPATH: common and rasa). Standard library only.
"""
import http.client
import time
from contextlib import nullcontext
from urllib.parse import urlsplit

from ari_servers import execution
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import config

# hop-by-hop headers: they belong to one connection, not forwarded
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


class RasaProxy(HTTPService):

    def __init__(self, cfg: config.RasaProxyConfig, exclusive):
        super().__init__("rasa_proxy", cfg.host, cfg.port, exclusive)
        self.cfg = cfg
        upstream = urlsplit(cfg.upstream)
        if upstream.scheme != "http" or not upstream.hostname:
            raise ValueError(f"upstream must be http://host:port, got {cfg.upstream}")
        self.upstream_host, self.upstream_port = upstream.hostname, upstream.port or 80
        self.fallback = self.forward

    def health(self) -> dict:
        try:
            status, _, _ = self._request("GET", "/", b"", {})
            ok = status < 500
        except OSError:
            ok = False
        return {"upstream": self.cfg.upstream, "upstream_ok": ok}

    def _request(self, method, path, body, headers):
        connection = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=self.cfg.timeout)
        try:
            connection.request(method, path, body=body or None, headers=headers)
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()

    def forward(self, request):
        length = int(request.headers.get("Content-Length", "0") or 0)
        body = request.read_body(self.cfg.max_body_mb * 1024 * 1024) if length else b""
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        method = request.command
        lock = self.execution.hold() if self.cfg.exclusive else nullcontext(0.0)
        with lock as waited:
            t_forward = time.perf_counter()
            try:
                status, reply_headers, reply = self._request(method, request.path, body, headers)
            except OSError as e:
                raise HTTPError(502, f"Rasa not reachable at {self.cfg.upstream}: {e}")
            t_end = time.perf_counter()
        mark("rasa_proxy", "done", client=request.client_address[0], method=method, path=request.route_path,
             status=status, queue_s=round(waited, 4), rasa_s=round(t_end - t_forward, 4),
             total_s=round(t_end - request.t_in, 4))
        passed = {k: v for k, v in reply_headers if k.lower() not in HOP_BY_HOP and k.lower() != "content-type"}
        content_type = next((v for k, v in reply_headers if k.lower() == "content-type"), "application/json")
        request.reply_raw(status, reply, content_type, passed)
        return None  # replied


def main(argv=None):
    setup_logging()
    RasaProxy(config.load(argv), execution.from_config()).serve_forever()


if __name__ == "__main__":
    main()

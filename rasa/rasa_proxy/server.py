"""RASA server of the architecture: a proxy in front of Rasa, so Rasa follows "no concurrent executions" too.

Rasa has its own HTTP API (rasa run --enable-api): the clients of the demo keep using it unchanged
(/webhooks/rest/webhook, /conversations/<id>/tracker/events, /model/parse, /), on the ports of this proxy.
One listener per route of RasaProxyConfig.routes (one per bot): every request is forwarded as it is to the bot of
that port, holding the execution lock of the servers, shared by all the listeners.

  GET /health   {"status": "ok", "service": "rasa_proxy_<port>", "upstream": ..., "upstream_ok": ...}
  any other     forwarded to the bot; 502 if it does not answer

Run: python -m rasa_proxy [--routes ...] (PYTHONPATH: common and rasa). Standard library only.
"""
import http.client
import threading
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

    def __init__(self, cfg: config.RasaProxyConfig, port: int, upstream_url: str, exclusive):
        super().__init__(f"rasa_proxy_{port}", cfg.host, port, exclusive)
        self.cfg, self.upstream_url = cfg, upstream_url
        upstream = urlsplit(upstream_url)
        if upstream.scheme != "http" or not upstream.hostname:
            raise ValueError(f"upstream must be http://host:port, got {upstream_url}")
        self.upstream_host, self.upstream_port = upstream.hostname, upstream.port or 80
        self.fallback = self.forward

    def health(self) -> dict:
        try:
            status, _, _ = self._request("GET", "/", b"", {})
            ok = status < 500
        except OSError:
            ok = False
        return {"upstream": self.upstream_url, "upstream_ok": ok}

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
                raise HTTPError(502, f"Rasa not reachable at {self.upstream_url}: {e}")
            t_end = time.perf_counter()
        mark("rasa_proxy", "done", port=self.port, client=request.client_address[0], method=method, path=request.route_path,
             status=status, queue_s=round(waited, 4), rasa_s=round(t_end - t_forward, 4),
             total_s=round(t_end - request.t_in, 4))
        passed = {k: v for k, v in reply_headers if k.lower() not in HOP_BY_HOP and k.lower() != "content-type"}
        content_type = next((v for k, v in reply_headers if k.lower() == "content-type"), "application/json")
        request.reply_raw(status, reply, content_type, passed)
        return None  # replied


def main(argv=None):
    setup_logging()
    cfg = config.load(argv)
    exclusive = execution.from_config()  # one lock for all the listeners
    proxies = [RasaProxy(cfg, port, upstream, exclusive) for port, upstream in cfg.parsed_routes()]
    threads = [threading.Thread(target=proxy.serve_forever, name=proxy.name, daemon=True) for proxy in proxies]
    for thread in threads:
        thread.start()
    # a listener that stops (e.g. port already in use) stops the process: the container restarts it
    while all(thread.is_alive() for thread in threads):
        threads[0].join(1.0)
    raise SystemExit("a listener of the RASA proxy stopped")


if __name__ == "__main__":
    main()

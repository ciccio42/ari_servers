"""LLM (+RAG) over the network: generation of the answers of the demo (no ROS). SKELETON.

The API, the configuration and the exclusive execution are in place; backend.py and rag.py are still to be ported
from PepperDemo, so the server starts, /health answers and /chat replies 501 until then
(--skip-load to start it without loading the models).

  GET  /health  {"status": "ok", "service": "llm", "model": ..., "backend": ..., "rag": ..., "loaded": ...}
  POST /chat    {"messages": [{"role": "user", "content": "..."}], "rag": true, "max_new_tokens": 350,
                 "temperature": 0.0}  or  {"prompt": "...", "system": "..."}
                reply: {"text", "chunks": [{"text", "source", "score"}], "rag_s", "generate_s", "queue_s", "total_s"}

Run: python -m llm_server [--model ...] [--port ...] (PYTHONPATH: common and llm).
"""
import argparse
import time
from dataclasses import asdict

from ari_servers import execution
from ari_servers.config import workspace_path
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import backend, config, rag


class LLMService(HTTPService):

    def __init__(self, cfg: config.LLMConfig, exclusive, load_models: bool = True):
        super().__init__("llm_server", cfg.host, cfg.port, exclusive)
        self.cfg = cfg
        self.backend = backend.create(cfg)
        self.retriever = rag.FaissRetriever(cfg) if cfg.rag else None
        self.system_prompt = workspace_path(cfg.system_prompt_file).read_text() if cfg.system_prompt_file else None
        self.loaded = False
        if load_models:
            self.backend.load()
            if self.retriever:
                self.retriever.load()
            self.loaded = True
        self.add_route("POST", "/chat", self.chat)

    def health(self) -> dict:
        return {**self.backend.describe(), "rag": self.cfg.rag, "loaded": self.loaded}

    def chat(self, request) -> dict:
        data = request.read_json(self.cfg.max_body_mb * 1024 * 1024)
        messages = data.get("messages") or [{"role": "user", "content": data.get("prompt", "")}]
        if not messages[-1].get("content"):
            raise HTTPError(400, "empty request: give messages or prompt")
        system = data.get("system", self.system_prompt)
        if system and messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system}, *messages]
        use_rag = bool(data.get("rag", self.cfg.rag))
        if use_rag and self.retriever is None:
            raise HTTPError(400, "RAG disabled in the configuration of the server")
        if not self.loaded:
            raise NotImplementedError("models not loaded (server started with --skip-load)")
        with self.execution.hold() as waited:
            t_rag = time.perf_counter()
            chunks = self.retriever.retrieve(messages[-1]["content"], self.cfg.top_k) if use_rag else []
            if chunks:
                context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
                messages = [*messages[:-1], {"role": "user",
                                             "content": f"Contesto:\n{context}\n\nDomanda: {messages[-1]['content']}"}]
            t_generate = time.perf_counter()
            text = self.backend.generate(messages, int(data.get("max_new_tokens", self.cfg.max_new_tokens)),
                                         float(data.get("temperature", self.cfg.temperature)))
            t_end = time.perf_counter()
        result = {"text": text, "chunks": [asdict(c) for c in chunks], "rag_s": round(t_generate - t_rag, 4),
                  "generate_s": round(t_end - t_generate, 4), "queue_s": round(waited, 4),
                  "total_s": round(t_end - request.t_in, 4)}
        mark("llm_server", "done", client=request.client_address[0], rag=use_rag, chunks=len(chunks),
             words=len(text.split()), rag_s=result["rag_s"], generate_s=result["generate_s"],
             queue_s=result["queue_s"], total_s=result["total_s"])
        return result


def main(argv=None):
    setup_logging()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--skip-load", action="store_true", help="start without loading the models (/chat replies 501)")
    known, rest = pre.parse_known_args(argv)
    LLMService(config.load(rest), execution.from_config(), load_models=not known.skip_load).serve_forever()


if __name__ == "__main__":
    main()

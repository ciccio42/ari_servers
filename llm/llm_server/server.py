"""LLM (+RAG) server: the answers of the demo over HTTP, for the custom actions of Rasa and any other client (no ROS).

The generation is done by a backend (backends/: ChatGPT, local transformers, echo for the tests), picked by
llm.backend. The server keeps the history of every conversation, adds the system prompt and, for the backends
without native RAG, the chunks of its retriever (rag.py). One request at a time among all the servers of the
machine (execution lock), during the generation.

  GET  /health  {"status": "ok", "service": "llm_server", "backend", "model", "native_rag", "vision", "rag",
                 "loaded", "conversations"}
  POST /chat    conversation with history:
                  {"conversation_id": "...", "message": "...", "image_base64": "<jpeg>" (optional),
                   "system": "..." (optional: default the prompt of llm.system_prompt_file), "rag": true|false
                   (optional: default llm.rag), "temperature", "max_new_tokens" (optional)}
                one-shot, without history:
                  {"messages": [{"role", "content"}...]}  or  {"prompt": "...", "system": "..."}
                reply: {"text", "model", "served_model", "rag", "chunks": [{"text", "source", "score"}],
                        "conversation_id", "history", "queue_s", "rag_s", "generate_s", "total_s"}
  POST /reset   {"conversation_id": "..."}: forgets its history. Reply {"reset": true|false}

Errors: 400 bad request, 501 backend not implemented or models not loaded, 502 the backend did not answer
(e.g. OpenAI unreachable or key refused), 500 other failures. The server keeps running.

Run: python -m llm_server [--backend ...] [--model ...] [--port ...] (PYTHONPATH: common and llm).
"""
import argparse
import time
from dataclasses import asdict

from ari_servers import execution
from ari_servers.config import workspace_path
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import backends, config, rag
from .conversation import Conversations


class LLMService(HTTPService):

    def __init__(self, cfg: config.LLMConfig, exclusive, load_models: bool = True):
        super().__init__("llm_server", cfg.host, cfg.port, exclusive)
        self.cfg = cfg
        self.backend = backends.create(cfg)
        # the retriever of the server only for the backends that do not search the documents themselves
        self.retriever = rag.FaissRetriever(cfg) if cfg.rag and not self.backend.native_rag else None
        self.system_prompt = None
        if cfg.system_prompt_file:
            self.system_prompt = workspace_path(cfg.system_prompt_file).read_text().strip().replace(
                "{max_char}", str(cfg.max_char))
        self.conversations = Conversations(cfg.history_length, cfg.max_conversations)
        self.loaded = False
        if load_models:
            self.backend.load()
            if self.retriever:
                self.retriever.load()
            self.loaded = True
        self.add_route("POST", "/chat", self.chat)
        self.add_route("POST", "/reset", self.reset)

    def health(self) -> dict:
        return {**self.backend.describe(), "rag": self.cfg.rag, "loaded": self.loaded,
                "conversations": len(self.conversations)}

    # --- requests

    def _request_message(self, data: dict) -> dict:
        text = str(data.get("message") or "").strip()
        if not text:
            raise HTTPError(400, "empty request: give message (with conversation_id), messages or prompt")
        image = data.get("image_base64")
        if not image:
            return {"role": "user", "content": text}
        if not self.backend.vision:
            raise HTTPError(400, f"backend {self.backend.name} does not accept images")
        return {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]}

    def chat(self, request) -> dict:
        data = request.read_json(self.cfg.max_body_mb * 1024 * 1024)
        conversation_id = data.get("conversation_id")
        if conversation_id is not None:
            conversation_id = str(conversation_id)
            pending = self.conversations.pending(conversation_id, self._request_message(data))
        elif data.get("messages"):
            pending = list(data["messages"])
        else:
            pending = [{"role": "user", "content": str(data.get("prompt") or data.get("message") or "")}]
        if not backends.text_of(pending[-1]).strip():
            raise HTTPError(400, "empty request")
        system = data.get("system") or self.system_prompt
        if pending[0].get("role") == "system":
            system, pending = pending[0]["content"], pending[1:]
        if not pending:
            raise HTTPError(400, "no request after the system prompt")
        use_rag = self.cfg.rag if data.get("rag") is None else bool(data["rag"])
        if use_rag and not self.backend.native_rag and self.retriever is None:
            raise HTTPError(400, "RAG disabled in the configuration of the server (llm.rag)")
        if not self.loaded:
            raise NotImplementedError("models not loaded (server started with --skip-load)")
        temperature = float(data.get("temperature", self.cfg.temperature))
        max_new_tokens = int(data.get("max_new_tokens", self.cfg.max_new_tokens))

        with self.execution.hold() as waited:
            t_rag = time.perf_counter()
            chunks, messages = [], list(pending)
            if use_rag and not self.backend.native_rag:
                chunks = self.retriever.retrieve(backends.text_of(pending[-1]), self.cfg.top_k)
                if chunks:
                    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
                    question = backends.text_of(pending[-1])
                    messages[-1] = {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {question}"}
            if system:
                messages = [{"role": "system", "content": system}, *messages]
            t_generate = time.perf_counter()
            try:
                reply = self.backend.generate(messages, rag=use_rag and self.backend.native_rag,
                                              temperature=temperature, max_new_tokens=max_new_tokens,
                                              timeout=self.cfg.timeout)
            except backends.BackendError as e:
                mark("llm_server", "failed", client=request.client_address[0], backend=self.backend.name, error=e)
                raise HTTPError(502, str(e))
            t_end = time.perf_counter()
        if conversation_id is not None:
            self.conversations.commit(conversation_id, pending, reply.text)

        result = {"text": reply.text, "model": reply.model, "served_model": reply.served_model,
                  "rag": use_rag, "chunks": [asdict(c) for c in chunks], "conversation_id": conversation_id,
                  "history": len(pending) - 1, "queue_s": round(waited, 4), "rag_s": round(t_generate - t_rag, 4),
                  "generate_s": round(t_end - t_generate, 4), "total_s": round(t_end - request.t_in, 4)}
        mark("llm_server", "done", client=request.client_address[0], backend=self.backend.name, model=reply.model,
             rag=use_rag, history=result["history"], words=len(reply.text.split()), queue_s=result["queue_s"],
             rag_s=result["rag_s"], generate_s=result["generate_s"], total_s=result["total_s"])
        return result

    def reset(self, request) -> dict:
        data = request.read_json()
        if data.get("conversation_id") is None:
            raise HTTPError(400, "give conversation_id")
        return {"reset": self.conversations.reset(str(data["conversation_id"]))}


def main(argv=None):
    setup_logging()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--skip-load", action="store_true", help="start without loading the backend (/chat replies 501)")
    known, rest = pre.parse_known_args(argv)
    LLMService(config.load(rest), execution.from_config(), load_models=not known.skip_load).serve_forever()


if __name__ == "__main__":
    main()

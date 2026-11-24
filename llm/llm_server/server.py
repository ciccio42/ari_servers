"""LLM (+RAG) server: the answers of the demo over HTTP, for the custom actions of Rasa and any other client (no ROS).

The generation is done by a backend (backends/: ChatGPT, local transformers, echo for the tests), picked by
llm.backend. The server keeps the history of every conversation, adds the system prompt and, for the backends
without native RAG, the chunks of its retriever (rag.py). One request at a time among all the servers of the
machine (execution lock), during the generation.

  GET  /health  {"status": "ok", "service": "llm_server", "backend", "model", "native_rag", "vision", "rag",
                 "loaded", "conversations"}
  POST /image   body: the image as bytes (Content-Type image/webp, image/jpeg or image/png), not base64: 25 % fewer
                bytes than inside a JSON. Kept in memory for IMAGE_TTL_S, then forgotten. Optional header X-Image-Id
                (32 hex characters): the id chosen by the client, so it can send the id elsewhere (e.g. in a slot of
                Rasa) while the upload is still running. Reply {"image_id", "bytes", "mime", "receive_s",
                "total_s"}; receive_s is the read of the body (the upload, network)
  POST /chat    conversation with history:
                  {"conversation_id": "...", "message": "...", "image_id": "<id of POST /image>" or
                   "image_base64": "<jpeg or webp>" (optional),
                   "system": "..." (optional: default the prompt of llm.system_prompt_file), "rag": true|false
                   (optional: default llm.rag) or the name of a RAG server of llm.rag_servers (context, system prompt
                   and model of that RAG, see ari_servers/rag.py), "temperature", "max_new_tokens" (optional)}
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
import dataclasses
import json
import re
import threading
import time
import uuid
from base64 import b64decode, b64encode
from dataclasses import asdict

from ari_servers import execution
from ari_servers.config import workspace_path
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import backends, config, rag
from .conversation import Conversations


IMAGE_TTL_S = 600   # an uploaded image is used by the answer that follows within seconds
IMAGE_MAX = 32      # images kept at most (the oldest go first)
IMAGE_TYPES = {"image/webp", "image/jpeg", "image/png"}


class ImageStore:
    """Images uploaded with POST /image, by id, for the requests to /chat that follow (thread safe)."""

    def __init__(self, ttl_s: float = IMAGE_TTL_S, max_images: int = IMAGE_MAX):
        self.ttl_s, self.max_images = ttl_s, max_images
        self._images = {}  # id -> (data, mime, time of the upload)
        self._lock = threading.Lock()

    def _expire(self, now: float):
        for key in [k for k, (_, _, t) in self._images.items() if now - t > self.ttl_s]:
            del self._images[key]
        while len(self._images) > self.max_images:
            del self._images[min(self._images, key=lambda k: self._images[k][2])]

    def put(self, data: bytes, mime: str, image_id: str = None) -> str:
        image_id = image_id or uuid.uuid4().hex
        with self._lock:
            now = time.monotonic()
            self._images[image_id] = (data, mime, now)
            self._expire(now)
        return image_id

    def get(self, image_id: str):
        with self._lock:
            self._expire(time.monotonic())
            found = self._images.get(image_id)
        return (found[0], found[1]) if found else None

    def __len__(self):
        return len(self._images)


def image_mime(data: bytes) -> str:
    """Type of an image from its first bytes (the base64 images of the old clients carry no type)."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


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
        self.rag_servers = rag.RagServers(cfg.rag_servers, cfg.rag_timeout)
        self.rag_backends = {}  # (backend, model, options) -> backend of the generator of a RAG server
        self._rag_backends_lock = threading.Lock()
        self.loaded = False
        if load_models:
            self.backend.load()
            if self.retriever:
                self.retriever.load()
            self.loaded = True
        self.images = ImageStore()
        self.add_route("POST", "/chat", self.chat)
        self.add_route("POST", "/image", self.upload_image)
        self.add_route("POST", "/reset", self.reset)

    def health(self) -> dict:
        return {**self.backend.describe(), "rag": self.cfg.rag, "rag_servers": self.rag_servers.urls,
                "loaded": self.loaded,
                "conversations": len(self.conversations), "images": len(self.images)}

    # --- requests

    def _request_message(self, data: dict) -> dict:
        text = str(data.get("message") or "").strip()
        if not text:
            raise HTTPError(400, "empty request: give message (with conversation_id), messages or prompt")
        image_id, image = data.get("image_id"), data.get("image_base64")
        if not image_id and not image:
            return {"role": "user", "content": text}
        if not self.backend.vision:
            raise HTTPError(400, f"backend {self.backend.name} does not accept images")
        if image_id:
            found = self.images.get(str(image_id))
            if found is None:
                raise HTTPError(400, f"unknown or expired image_id {image_id} (POST /image, kept {IMAGE_TTL_S} s)")
            raw, mime = found
            image = b64encode(raw).decode()
        else:
            try:
                mime = image_mime(b64decode(image[:24]))
            except ValueError:
                mime = "image/jpeg"
        return {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}}]}

    def upload_image(self, request) -> dict:
        mime = (request.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        image_id = (request.headers.get("X-Image-Id") or "").strip().lower() or None
        if image_id is not None and not re.fullmatch(r"[0-9a-f]{32}", image_id):
            raise HTTPError(400, "X-Image-Id must be 32 hexadecimal characters")
        data = request.read_body(self.cfg.max_body_mb * 1024 * 1024)
        t_received = time.perf_counter()
        if mime not in IMAGE_TYPES:
            mime = image_mime(data)
        image_id = self.images.put(data, mime, image_id)
        t_end = time.perf_counter()
        result = {"image_id": image_id, "bytes": len(data), "mime": mime,
                  "receive_s": round(t_received - request.t_in, 4), "total_s": round(t_end - t_received, 4)}
        mark("llm_server", "image", client=request.client_address[0], bytes=len(data), mime=mime,
             receive_s=result["receive_s"], total_s=result["total_s"])
        return result

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
        rag_server = self._rag_server(data.get("rag"))
        if rag_server is not None:
            return self._chat_rag_server(request, data, rag_server, conversation_id, pending)
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

    # --- RAG servers (rag/<name>): "rag": "<name>"

    def _rag_server(self, value):
        """The name of a RAG server when the request gives one in "rag", None for true/false/absent."""
        if not isinstance(value, str) or value.strip().lower() in ("", "true", "false", "1", "0", "yes", "no"):
            return None
        name = value.strip()
        if name not in self.rag_servers:
            raise HTTPError(400, f"unknown RAG server {name}: {self.rag_servers.names()} (llm.rag_servers)")
        return name

    def _generator_backend(self, generator: dict):
        """The backend of the generator declared by a RAG server, created once and kept."""
        options = generator.get("options") or {}
        key = (generator.get("backend"), generator.get("model"), json.dumps(options, sort_keys=True))
        with self._rag_backends_lock:
            if key not in self.rag_backends:
                cfg = dataclasses.replace(self.cfg, backend=generator.get("backend") or self.cfg.backend,
                                          model=generator.get("model") or self.cfg.model)
                try:
                    backend = backends.create(cfg, options)
                except ValueError as e:
                    raise HTTPError(502, f"generator of the RAG server: {e}")
                backend.load()
                self.rag_backends[key] = backend
            return self.rag_backends[key]

    def _chat_rag_server(self, request, data: dict, name: str, conversation_id, pending: list) -> dict:
        """Context and system prompt from the RAG server, answer with its model (outside the lock: the RAG server
        takes it during its retrieval; this server takes it during the generation)."""
        if not self.loaded:
            raise NotImplementedError("models not loaded (server started with --skip-load)")
        question = backends.text_of(pending[-1]).strip()
        t_rag = time.perf_counter()
        try:
            found = self.rag_servers.retrieve(name, question)
        except rag.RagServerError as e:
            mark("llm_server", "failed", client=request.client_address[0], rag=name, error=e)
            raise HTTPError(502, str(e))
        generator = found.get("generator") or {}
        backend = self._generator_backend(generator)
        waited = 0.0
        t_generate = time.perf_counter()
        if found.get("answer"):  # the RAG answers without the model (e.g. nothing retrieved)
            text, model, served = found["answer"], generator.get("model"), None
        else:
            messages = [*pending[:-1], {"role": "user", "content": found.get("user") or question}]
            system = found.get("system") or data.get("system") or self.system_prompt
            if system:
                messages = [{"role": "system", "content": system}, *messages]
            temperature = float(data.get("temperature", generator.get("temperature", self.cfg.temperature)))
            max_new_tokens = int(data.get("max_new_tokens", generator.get("max_new_tokens", self.cfg.max_new_tokens)))
            with self.execution.hold() as waited:
                t_generate = time.perf_counter()
                try:
                    reply = backend.generate(messages, rag=False, temperature=temperature,
                                             max_new_tokens=max_new_tokens, timeout=self.cfg.timeout)
                except backends.BackendError as e:
                    mark("llm_server", "failed", client=request.client_address[0], backend=backend.name, rag=name,
                         error=e)
                    raise HTTPError(502, str(e))
            text, model, served = reply.text, reply.model, reply.served_model
            for pattern in found.get("drop_patterns") or []:
                text = re.sub(pattern, "", text)
            text = text.strip()
        t_end = time.perf_counter()
        if conversation_id is not None:  # the history keeps the question, not the documents
            self.conversations.commit(conversation_id, pending, text)

        result = {"text": text, "model": model, "served_model": served, "rag": name,
                  "chunks": found.get("chunks") or [], "conversation_id": conversation_id,
                  "history": len(pending) - 1, "queue_s": round(waited + found.get("queue_s", 0.0), 4),
                  "rag_s": round(t_generate - t_rag, 4), "generate_s": round(t_end - t_generate, 4),
                  "total_s": round(t_end - request.t_in, 4),
                  "rag_info": {k: found.get(k) for k in ("filters", "registers", "sources") if k in found}}
        mark("llm_server", "done", client=request.client_address[0], backend=backend.name, model=model, rag=name,
             history=result["history"], words=len(text.split()), queue_s=result["queue_s"], rag_s=result["rag_s"],
             generate_s=result["generate_s"], total_s=result["total_s"])
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

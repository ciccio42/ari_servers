"""Ollama backend: a local model served by Ollama (ollama/docker/run.bash, models in models/ollama), over its
native API (POST /api/chat), the only one that accepts the context length (num_ctx) per request.

It is the generator of the local RAGs: rag/department_office answers with qwen3.6:35b-a3b-q8_0, as in the
evaluations of rag_unisa. The model is LLMConfig.model; the other parameters are in the section `ollama` of
config/servers.yaml (env OLLAMA_<FIELD>), the RAG servers can override them (options of their generator).
Standard library only.
"""
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List

from ari_servers import config
from ari_servers.latency_log import mark

from .base import BackendError, ChatBackend, Message, Reply, has_image, text_of

log = logging.getLogger("ollama")


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11500"
    num_ctx: int = 16384  # tokens of context: Ollama would size it on the free memory (262144 on the Spark)
    keep_alive: str = "2h"  # the model stays loaded between the requests
    think: bool = False  # explicit reasoning of the models that have it (qwen3): slower, not needed here


class OllamaBackend(ChatBackend):
    name = "ollama"

    def __init__(self, cfg, options: dict = None):
        super().__init__(cfg)
        self.ollama = config.load(OllamaConfig, "ollama", "OLLAMA", argv=[])
        self.options = dict(options or {})  # num_ctx, think, keep_alive of a RAG: they override the section ollama

    def _option(self, name):
        return self.options.get(name, getattr(self.ollama, name))

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(f"{self.ollama.url.rstrip('/')}{path}", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return json.loads(reply.read())

    def available(self, timeout: float = 5.0):
        """Models present on the Ollama server, None if it does not answer."""
        try:
            with urllib.request.urlopen(f"{self.ollama.url.rstrip('/')}/api/tags", timeout=timeout) as reply:
                return sorted(m.get("name", "") for m in json.loads(reply.read()).get("models", []))
        except (OSError, ValueError):
            return None

    def load(self) -> None:
        models = self.available()
        if models is None:
            log.warning("Ollama does not answer at %s: start it with bash ollama/docker/run.bash start", self.ollama.url)
        elif self.cfg.model not in models and f"{self.cfg.model}:latest" not in models:
            log.warning("model %s not on %s (present: %s): bash ollama/docker/run.bash pull %s",
                        self.cfg.model, self.ollama.url, ", ".join(models) or "none", self.cfg.model)
        else:
            log.info("Ollama model %s at %s, num_ctx %s", self.cfg.model, self.ollama.url, self._option("num_ctx"))

    def describe(self) -> dict:
        return {**super().describe(), "url": self.ollama.url, "num_ctx": self._option("num_ctx")}

    def generate(self, messages: List[Message], *, rag: bool, temperature: float, max_new_tokens: int,
                 timeout: float) -> Reply:
        if any(has_image(m) for m in messages):
            raise BackendError("the ollama backend does not accept images")
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": m["role"], "content": text_of(m)} for m in messages],
            "stream": False,
            "think": bool(self._option("think")),
            "keep_alive": self._option("keep_alive"),
            "options": {"temperature": temperature, "num_predict": max_new_tokens,
                        "num_ctx": int(self._option("num_ctx"))},
        }
        start = time.time()
        mark("ollama", "request", model=self.cfg.model)
        for attempt in range(2):
            try:
                data = self._post("/api/chat", payload, timeout)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                # a model without reasoning refuses "think": once more without it, as rag_unisa
                if e.code == 400 and "think" in payload and attempt == 0:
                    payload.pop("think")
                    continue
                if e.code == 404:
                    raise BackendError(f"Ollama: model {self.cfg.model} not on {self.ollama.url} "
                                       f"(bash ollama/docker/run.bash pull {self.cfg.model}): {detail}")
                raise BackendError(f"Ollama: HTTP {e.code}: {detail}")
            except (OSError, ValueError) as e:  # connection refused, timeout, bad JSON
                raise BackendError(f"Ollama at {self.ollama.url}: {e}")
        text = ((data.get("message") or {}).get("content") or "").strip()
        mark("ollama", "reply", model=self.cfg.model, chars=len(text), prompt_tokens=data.get("prompt_eval_count"),
             tokens=data.get("eval_count"), elapsed_s=round(time.time() - start, 3))
        if not text:
            raise BackendError("Ollama: empty answer")
        return Reply(text=text, model=self.cfg.model, served_model=data.get("model"))

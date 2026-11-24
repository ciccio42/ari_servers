"""vLLM backend: a model served by vLLM (vllm/docker/run.bash) on its OpenAI-compatible API (POST /v1/chat/completions).

It is the generator of the high_level_unisa RAG: Qwen3.6-35B-A3B in NVFP4 (served as qwen3.6-nvfp4), with the final
configuration of the repository high-level-unisa (doc/resoconto_attivita_5.pdf, test_baseline/sistema.sh): temperature 0,
reasoning off (chat_template_kwargs enable_thinking false). The model is LLMConfig.model (the served name); the address and the
settings of the server are in the section `vllm` of config/servers.yaml (env VLLM_<FIELD>). Standard library only.
"""
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from ari_servers import config
from ari_servers.latency_log import mark

from .base import BackendError, ChatBackend, Message, Reply, has_image, text_of

log = logging.getLogger("vllm")


@dataclass
class VllmConfig:
    url: str = "http://localhost:8001"
    enable_thinking: bool = False  # chat_template_kwargs of the requests, as confronto_backend.py
    # settings of the server, read by vllm/docker/run.bash (sistema.sh of high-level-unisa): here only so that the
    # section `vllm` is one
    image: str = "vllm/vllm-openai:v0.31.0"
    cache_volume: str = "ari-vllm-cache-031"
    model: str = "nvidia/Qwen3.6-35B-A3B-NVFP4"
    revision: str = "main"
    hf_home: str = "models/vllm"
    served_model: str = "qwen3.6-nvfp4"
    gpu_memory_utilization: float = 0.40
    max_model_len: int = 131072
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192
    speculative_config: Optional[dict] = None
    prefix_caching: bool = True
    reserve_gb: int = 26


class VllmBackend(ChatBackend):
    name = "vllm"

    def __init__(self, cfg, options: dict = None):
        super().__init__(cfg)
        self.vllm = config.load(VllmConfig, "vllm", "VLLM", argv=[])
        self.options = dict(options or {})  # enable_thinking of a RAG: overrides the section vllm

    def available(self, timeout: float = 5.0):
        """Models served by vLLM, None if it does not answer."""
        try:
            with urllib.request.urlopen(f"{self.vllm.url.rstrip('/')}/v1/models", timeout=timeout) as reply:
                return [m.get("id") for m in json.loads(reply.read()).get("data", [])]
        except (OSError, ValueError):
            return None

    def load(self) -> None:
        models = self.available()
        if models is None:
            log.warning("vLLM does not answer at %s: load it with the orchestrator (or bash vllm/docker/run.bash start)",
                        self.vllm.url)
        elif self.cfg.model not in models:
            log.warning("model %s not served by vLLM at %s (served: %s)", self.cfg.model, self.vllm.url, models)
        else:
            log.info("vLLM model %s at %s", self.cfg.model, self.vllm.url)

    def describe(self) -> dict:
        return {**super().describe(), "url": self.vllm.url}

    def generate(self, messages: List[Message], *, rag: bool, temperature: float, max_new_tokens: int,
                 timeout: float) -> Reply:
        if any(has_image(m) for m in messages):
            raise BackendError("the vllm backend does not accept images")
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": m["role"], "content": text_of(m)} for m in messages],
            "temperature": temperature, "max_tokens": max_new_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": bool(self.options.get("enable_thinking",
                                                                              self.vllm.enable_thinking))},
        }
        request = urllib.request.Request(f"{self.vllm.url.rstrip('/')}/v1/chat/completions",
                                         data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        start = time.time()
        mark("vllm", "request", model=self.cfg.model)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                data = json.loads(reply.read())
        except urllib.error.HTTPError as e:
            raise BackendError(f"vLLM: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except (OSError, ValueError) as e:
            raise BackendError(f"vLLM at {self.vllm.url}: {e} (load it with the orchestrator)")
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        mark("vllm", "reply", model=self.cfg.model, chars=len(text), prompt_tokens=usage.get("prompt_tokens"),
             tokens=usage.get("completion_tokens"), elapsed_s=round(time.time() - start, 3))
        if not text.strip():
            raise BackendError("vLLM: empty answer")
        return Reply(text=text.strip(), model=self.cfg.model, served_model=data.get("model"))

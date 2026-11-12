"""Generation backends of the LLM server: the model is a parameter (LLMConfig.model, .backend).

ChatBackend is the interface the server uses; a new backend (vLLM, llama.cpp, TensorRT-LLM...) is a subclass
registered in BACKENDS. TODO: port ChatModel of PepperDemo (ros2_ws/src/trees/trees/llm_rag_utils/chatmodel.py)
into TransformersBackend.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from .config import LLMConfig

Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


class ChatBackend(ABC):

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    @abstractmethod
    def load(self) -> None:
        """Loads the model (cfg.model, cfg.model_dir) on the GPU: called once at the start of the server."""

    @abstractmethod
    def generate(self, messages: List[Message], max_new_tokens: int, temperature: float) -> str:
        """Answer of the model to the conversation `messages`."""

    def describe(self) -> dict:
        return {"backend": self.cfg.backend, "model": self.cfg.model}


class TransformersBackend(ChatBackend):
    """Hugging Face transformers, as ChatModel of PepperDemo."""

    def load(self) -> None:
        raise NotImplementedError("TransformersBackend.load: port ChatModel of PepperDemo (llm_rag_utils/chatmodel.py)")

    def generate(self, messages, max_new_tokens, temperature) -> str:
        raise NotImplementedError("TransformersBackend.generate")


BACKENDS = {"transformers": TransformersBackend}


def create(cfg: LLMConfig) -> ChatBackend:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend {cfg.backend} not available: {list(BACKENDS)}")
    return BACKENDS[cfg.backend](cfg)

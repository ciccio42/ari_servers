"""Generation backends of the LLM server: the abstract ChatBackend (base.py) and its implementations.
llm.backend in config/servers.yaml (or LLM_BACKEND, --backend) picks one."""
from .base import BackendError, ChatBackend, Message, Reply, has_image, text_of
from .echo import EchoBackend
from .openai_chat import ChatGPTBackend
from .transformers_local import TransformersBackend

BACKENDS = {backend.name: backend for backend in (ChatGPTBackend, TransformersBackend, EchoBackend)}


def create(cfg) -> ChatBackend:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend {cfg.backend} not available: {sorted(BACKENDS)}")
    return BACKENDS[cfg.backend](cfg)

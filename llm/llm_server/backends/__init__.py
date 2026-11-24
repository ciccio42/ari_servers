"""Generation backends of the LLM server: the abstract ChatBackend (base.py) and its implementations.
llm.backend in config/servers.yaml (or LLM_BACKEND, --backend) picks one."""
from .base import BackendError, ChatBackend, Message, Reply, has_image, text_of
from .echo import EchoBackend
from .ollama import OllamaBackend
from .openai_chat import ChatGPTBackend
from .transformers_local import TransformersBackend
from .vllm import VllmBackend

BACKENDS = {backend.name: backend for backend in (ChatGPTBackend, OllamaBackend, VllmBackend, TransformersBackend, EchoBackend)}


def create(cfg, options: dict = None) -> ChatBackend:
    """The backend cfg.backend with the model cfg.model; options: of the generator of a RAG (ollama only)."""
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend {cfg.backend} not available: {sorted(BACKENDS)}")
    if options:
        return BACKENDS[cfg.backend](cfg, options=options)
    return BACKENDS[cfg.backend](cfg)

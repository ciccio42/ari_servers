"""Interface of the generation backends of the LLM server. The reference is the ChatGPT backend (openai_chat.py),
ported from the custom actions of Rasa: a new backend (local transformers, vLLM, llama.cpp...) is a subclass of
ChatBackend registered in backends/__init__.py, and the server, its API and its clients do not change.

The backend is stateless: it answers a whole conversation. The server keeps the history of every conversation,
adds the system prompt and, for the backends without native_rag, the chunks of its own retriever.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# OpenAI chat format, the one of the reference backend:
#   {"role": "system" | "user" | "assistant", "content": "text"}
#   {"role": "user", "content": [{"type": "text", "text": ...},
#                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}
Message = Dict[str, Any]


class BackendError(Exception):
    """The backend could not answer (API error, timeout, model failure): the server replies 502."""


@dataclass
class Reply:
    text: str
    model: str  # the model asked for
    served_model: Optional[str] = None  # the model that answered, when the backend says it (e.g. gpt-4o-2024-08-06)
    rag: bool = False  # the backend searched the documents itself (native_rag)
    attempts: int = 1


class ChatBackend(ABC):
    name = "abstract"  # key of the backend in BACKENDS and in llm.backend of servers.yaml
    native_rag = False  # True: the backend searches the documents itself (OpenAI file_search); False: the server
    #                     gives it the chunks of its retriever (rag.py)
    vision = False  # True: it accepts images in the messages (image_url parts)

    def __init__(self, cfg):
        self.cfg = cfg  # LLMConfig

    @abstractmethod
    def load(self) -> None:
        """Prepares the backend (API client, model on the GPU): called once at the start of the server."""

    @abstractmethod
    def generate(self, messages: List[Message], *, rag: bool, temperature: float, max_new_tokens: int,
                 timeout: float) -> Reply:
        """Answer to `messages` (system prompt first, then the history, the last one is the request).
        rag: search the documents (only for native_rag backends). Raises BackendError when there is no answer."""

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.cfg.model, "native_rag": self.native_rag, "vision": self.vision}


def text_of(message: Message) -> str:
    """The text of a message, also when its content has image parts."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if part.get("type") == "text")


def has_image(message: Message) -> bool:
    content = message.get("content", "")
    return not isinstance(content, str) and any(part.get("type") == "image_url" for part in content)

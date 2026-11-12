"""Backend for the tests: it answers with the request, without models or keys. Not for the demo."""
from typing import List

from .base import ChatBackend, Message, Reply, has_image, text_of


class EchoBackend(ChatBackend):
    name = "echo"
    vision = True

    def load(self) -> None:
        pass

    def generate(self, messages: List[Message], *, rag, temperature, max_new_tokens, timeout) -> Reply:
        last = messages[-1]
        history = sum(1 for m in messages if m["role"] in ("user", "assistant")) - 1
        image = " [image]" if has_image(last) else ""
        return Reply(text=f"echo: {text_of(last)}{image} (history: {history})", model="echo")

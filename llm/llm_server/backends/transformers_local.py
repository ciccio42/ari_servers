"""Local model with Hugging Face transformers on the GPU of the Spark. TODO: port ChatModel of PepperDemo
(llm_rag_utils/chatmodel.py): model in models/llm/<name> (LLMConfig.model_dir), dtype LLMConfig.dtype.
The documents come from the retriever of the server (rag.py), so native_rag is False."""
from typing import List

from .base import ChatBackend, Message, Reply


class TransformersBackend(ChatBackend):
    name = "transformers"

    def load(self) -> None:
        raise NotImplementedError("TransformersBackend.load: port ChatModel of PepperDemo (llm_rag_utils/chatmodel.py)")

    def generate(self, messages: List[Message], *, rag, temperature, max_new_tokens, timeout) -> Reply:
        raise NotImplementedError("TransformersBackend.generate")

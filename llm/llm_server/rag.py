"""Retrieval of the LLM server: the chunks of the documents given to the model as context.

TODO: port Rag of PepperDemo (ros2_ws/src/trees/trees/llm_rag_utils/rag.py and retrieval.py: FAISS + BM25,
reranker, reciprocal rank fusion) into FaissRetriever; the index is built by prepare_rag.py of PepperDemo.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .config import LLMConfig


@dataclass
class Chunk:
    text: str
    source: str = ""  # document and section
    score: float = 0.0


class Retriever(ABC):

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    @abstractmethod
    def load(self) -> None:
        """Loads index, embedding model and reranker: called once at the start of the server."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """The `top_k` chunks most relevant to `query`, best first."""


class FaissRetriever(Retriever):

    def load(self) -> None:
        raise NotImplementedError("FaissRetriever.load: port Rag of PepperDemo (llm_rag_utils/rag.py)")

    def retrieve(self, query, top_k) -> List[Chunk]:
        raise NotImplementedError("FaissRetriever.retrieve")

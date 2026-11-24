"""Retrieval of the LLM server: the chunks of the documents given to the model as context.

RagServers: the RAG servers of rag/ (llm.rag_servers), asked with "rag": "<name>" in the requests.

TODO: port Rag of PepperDemo (ros2_ws/src/trees/trees/llm_rag_utils/rag.py and retrieval.py: FAISS + BM25,
reranker, reciprocal rank fusion) into FaissRetriever; the index is built by prepare_rag.py of PepperDemo.
"""
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List

from ari_servers.rag import Chunk  # noqa: F401 (also the format of the chunks of the RAG servers)

from .config import LLMConfig


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


class RagServerError(Exception):
    """The RAG server did not answer: the LLM server replies 502."""


class RagServers:
    """The RAG servers of llm.rag_servers ("<name>=<url>, ..."): POST /retrieve of ari_servers/rag.py."""

    def __init__(self, spec: str, timeout: float):
        self.timeout = timeout
        self.urls: Dict[str, str] = {}
        for item in (spec or "").split(","):
            if item.strip():
                name, _, url = item.partition("=")
                if not url.strip():
                    raise SystemExit(f"llm.rag_servers: {item!r} is not <name>=<url>")
                self.urls[name.strip()] = url.strip().rstrip("/")

    def __contains__(self, name: str) -> bool:
        return name in self.urls

    def names(self) -> List[str]:
        return sorted(self.urls)

    def retrieve(self, name: str, question: str, history: List[dict] = None) -> dict:
        return self._post(name, "/retrieve", {"question": question, "history": history or []})

    def postprocess(self, name: str, retrieval_id: str, answer: str) -> dict:
        return self._post(name, "/postprocess", {"retrieval_id": retrieval_id, "answer": answer})

    def _post(self, name: str, path: str, body: dict) -> dict:
        request = urllib.request.Request(f"{self.urls[name]}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as e:
            raise RagServerError(f"RAG server {name}: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except (OSError, ValueError) as e:
            raise RagServerError(f"RAG server {name} at {self.urls[name]}: {e} (not loaded: POST /load {{\"rag\": \"{name}\"}} of the orchestrator)")

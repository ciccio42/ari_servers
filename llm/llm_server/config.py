"""Configuration of the LLM (+RAG) server: section `llm` of config/servers.yaml, LLM_<FIELD>, --<field>."""
from dataclasses import dataclass
from typing import Optional

from ari_servers import config


@dataclass
class LLMConfig:
    host: str = "0.0.0.0"
    port: int = 8010
    backend: str = "openai"  # see backends.BACKENDS: openai (ChatGPT), transformers (local, to port), echo (tests)
    model: str = "gpt-4o"  # OpenAI model, or Hugging Face id for the local backends
    model_dir: Optional[str] = None  # local backends: default models/llm/<model name>
    dtype: str = "bfloat16"
    max_new_tokens: int = 350  # local backends, as service.py of PepperDemo
    temperature: float = 0.3  # 0: greedy. ChatGPT: only the non-reasoning models (gpt-4o, gpt-4o-mini)
    timeout: float = 60.0  # seconds for one answer, the new attempts of the backend included
    # default system prompt, when the request gives none; {max_char} becomes max_char. None: no system prompt
    system_prompt_file: Optional[str] = "llm/prompts/ari_system.txt"
    max_char: int = 250  # length of the answers asked by the default system prompt
    history_length: int = 4  # messages kept per conversation, the request included (even: 4 = 2 requests + 2 answers)
    max_conversations: int = 200  # the oldest are dropped
    # default of the requests that do not say it. ChatGPT: file_search on the OpenAI vector store (section openai);
    # local backends: the retriever of rag.py (rag_index, embedding_model, reranker_model, top_k)
    rag: bool = False
    rag_index: str = "models/rag/index"
    embedding_model: str = "google/embeddinggemma-300m"  # defaults of rag.py of PepperDemo
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 4
    max_body_mb: int = 16  # the requests can carry an image (base64)


def load(argv=None) -> LLMConfig:
    return config.load(LLMConfig, "llm", "LLM", argv, description="LLM (+RAG) HTTP server")

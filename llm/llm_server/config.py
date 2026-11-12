"""Configuration of the LLM (+RAG) server: section `llm` of config/servers.yaml, LLM_<FIELD>, --<field>."""
from dataclasses import dataclass
from typing import Optional

from ari_servers import config


@dataclass
class LLMConfig:
    host: str = "0.0.0.0"
    port: int = 8010
    model: str = "meta-llama/Llama-3.1-8B-Instruct"  # Hugging Face id
    model_dir: Optional[str] = None  # default: models/llm/<model name>
    backend: str = "transformers"  # see backend.BACKENDS
    dtype: str = "bfloat16"
    max_new_tokens: int = 350  # as service.py of PepperDemo
    temperature: float = 0.0  # 0: greedy
    system_prompt_file: Optional[str] = None  # text file; none: no system prompt
    rag: bool = True
    rag_index: str = "models/rag/index"
    embedding_model: str = "google/embeddinggemma-300m"  # defaults of rag.py of PepperDemo
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 4
    max_body_mb: int = 4


def load(argv=None) -> LLMConfig:
    return config.load(LLMConfig, "llm", "LLM", argv, description="LLM (+RAG) HTTP server")

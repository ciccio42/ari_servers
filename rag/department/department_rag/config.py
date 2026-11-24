"""Configuration of the department RAG: section rag_department of config/servers.yaml, RAG_DEPARTMENT_<FIELD>,
--<field>. The defaults are the ones of DIEM_RAG (diem_RAG_routered.py)."""
from dataclasses import dataclass

from ari_servers import config
from ari_servers.rag import RagConfig


@dataclass
class DepartmentConfig(RagConfig):
    port: int = 8032
    kb_dir: str = "rag_knowledge_base/department"
    vectorstore: str = "vectorstore"  # Chroma index in kb_dir
    # generation, done by the LLM server: Qwen3-30B-A3B-Instruct-2507 of the original (transformers, greedy, 4096
    # tokens), fp16 on Ollama
    generator_backend: str = "ollama"
    generator_model: str = "qwen3:30b-a3b-instruct-2507-fp16"
    temperature: float = 0.0
    max_new_tokens: int = 4096
    num_ctx: int = 16384  # the same for the generation and the calls of the RAG (domain filter, rewriting, router)
    ollama_url: str = "http://localhost:11500"
    # the original used a new empty history at every request ("HARDCODATO PER IL TEST, POI RIMETTI NORMALE"): true
    # gives the history of the conversation to the rewriting and to the answer, as the code intended
    use_history: bool = True


def load(argv=None) -> DepartmentConfig:
    return config.load(DepartmentConfig, "rag_department", "RAG_DEPARTMENT", argv,
                       description="RAG server of the DIEM department")

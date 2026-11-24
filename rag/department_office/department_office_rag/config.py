"""Configuration of the department_office RAG: section rag_department_office of config/servers.yaml,
RAG_DEPARTMENT_OFFICE_<FIELD>, --<field>. The defaults are the ones of ConfigRisposta of rag_unisa."""
from dataclasses import dataclass

from ari_servers import config
from ari_servers.rag import RagConfig


@dataclass
class DepartmentOfficeConfig(RagConfig):
    port: int = 8030
    kb_dir: str = "rag_knowledge_base/department_office"  # qdrant/ (index) and registers.json
    # generation, done by the LLM server with these values (the model of the evaluations of rag_unisa)
    generator_backend: str = "ollama"
    generator_model: str = "qwen3.6:35b-a3b-q8_0"
    temperature: float = 0.1
    max_new_tokens: int = 1500  # 800 (rag_unisa) cut the attendance lists
    num_ctx: int = 16384  # context of Ollama: 30000 characters of documents are 9000-11400 tokens
    # Ollama, for the planner of the registers (tool calls on the questions over many meetings)
    ollama_url: str = "http://localhost:11500"
    timeout: float = 300.0
    # retrieval
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # empty: no reranking
    device: str = "cuda"
    top_k: int = 14  # passages in the context
    candidates: int = 40  # passages given to the reranker
    mode: str = "ibrido"  # ibrido (dense + sparse), denso, sparso
    expansion: int = 10  # nearby items added to the context (0: off)
    context_max_chars: int = 30000
    use_registers: bool = True
    latest_versions: bool = True  # without a time in the question, the newest edition of a reissued document
    complete_sentences: bool = True  # answers as a complete sentence


def load(argv=None) -> DepartmentOfficeConfig:
    return config.load(DepartmentOfficeConfig, "rag_department_office", "RAG_DEPARTMENT_OFFICE", argv,
                       description="RAG server of the department resolutions")

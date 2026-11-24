"""Configuration of the high_level_unisa RAG: section rag_high_level_unisa of config/servers.yaml,
RAG_HIGH_LEVEL_UNISA_<FIELD>, --<field>. The defaults are the final configuration of high-level-unisa
(versione_finale/LEGGIMI.md): they become the environment variables read by utils/pipeline.py (see ENVIRONMENT)."""
from dataclasses import dataclass

from ari_servers import config
from ari_servers.rag import RagConfig


@dataclass
class HighLevelUnisaConfig(RagConfig):
    port: int = 8031
    kb_dir: str = "rag_knowledge_base/high_level_unisa"  # diem_vectorstore/, anagrafe.json, registro_strutture.json
    vectorstore: str = "diem_vectorstore"  # Chroma index in kb_dir
    # generation, done by the LLM server with these values (pipeline.py: ChatOllama, temperature 0, 4096 tokens)
    generator_backend: str = "ollama"
    generator_model: str = "qwen3.6:35b-a3b"
    temperature: float = 0.0
    max_new_tokens: int = 4096
    # context of Ollama, the same for the generation and the calls of the RAG (rewriting, list mode): the longest
    # prompt measured by high-level-unisa is 17953 tokens
    num_ctx: int = 32768
    ollama_url: str = "http://localhost:11500"  # rewriting of the question and list mode
    thinking: bool = False  # OLLAMA_ENABLE_THINKING=false of the final configuration
    # switches of pipeline.py (final configuration)
    person_pipeline: bool = True
    struct_path: bool = True
    struct_attributes: bool = True
    list_mode: bool = True
    list_mode_top_n: int = 20
    name_check: bool = True
    rerank_top_n: int = 10
    bm25_k: int = 30
    context_order: str = "coda"
    domain_guardrail_mode: str = "permissivo"  # permissivo (rules on the greetings), llm, off
    benchmark_date: str = ""  # dd/mm/yyyy: the date of the prompt frozen, for the tests
    # cache of the BM25 retrievers and of the common words, relative to kb_dir (empty: none): built at the first start
    # (about 1 GB), read at the next ones; built again by itself when the index changes
    bm25_cache_dir: str = "cache"


# field -> environment variable of utils/pipeline.py
ENVIRONMENT = {
    "generator_model": "OLLAMA_MODEL_ID", "ollama_url": "OLLAMA_BASE_URL", "num_ctx": "OLLAMA_NUM_CTX",
    "thinking": "OLLAMA_ENABLE_THINKING", "person_pipeline": "ENABLE_PERSON_PIPELINE",
    "struct_path": "ENABLE_STRUCT_PATH", "struct_attributes": "ENABLE_STRUCT_ATTRIBUTES",
    "list_mode": "ENABLE_LIST_MODE", "list_mode_top_n": "LIST_MODE_TOP_N", "name_check": "ENABLE_NAME_CHECK",
    "rerank_top_n": "RERANK_TOP_N", "bm25_k": "BM25_K", "context_order": "CONTEXT_ORDER",
    "domain_guardrail_mode": "DOMAIN_GUARDRAIL_MODE", "benchmark_date": "BENCHMARK_DATE",
}


def load(argv=None) -> HighLevelUnisaConfig:
    return config.load(HighLevelUnisaConfig, "rag_high_level_unisa", "RAG_HIGH_LEVEL_UNISA", argv,
                       description="RAG server of the University of Salerno (Ateneo)")

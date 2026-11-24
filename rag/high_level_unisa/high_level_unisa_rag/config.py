"""Configuration of the high_level_unisa RAG: section rag_high_level_unisa of config/servers.yaml,
RAG_HIGH_LEVEL_UNISA_<FIELD>, --<field>. The defaults are the final configuration of high-level-unisa
(doc/resoconto_attivita_5.pdf, test_baseline/sistema.sh: PIPELINE_ENV and the CONFIGURAZIONE of telegram_bot_vllm.py):
they become the environment variables read by utils/pipeline.py (see ENVIRONMENT)."""
from dataclasses import dataclass

from ari_servers import config
from ari_servers.rag import RagConfig


@dataclass
class HighLevelUnisaConfig(RagConfig):
    port: int = 8031
    kb_dir: str = "rag_knowledge_base/high_level_unisa"  # diem_vectorstore/, anagrafe.json, registro_strutture.json
    vectorstore: str = "diem_vectorstore"  # Chroma index in kb_dir
    # generation, done by the LLM server with these values (pipeline.py: temperature 0, 4096 tokens). vllm: the engine
    # of the final configuration (NVFP4, served as qwen3.6-nvfp4), also for the calls of the RAG; ollama:
    # qwen3.6:35b-a3b (Q4), the model of versione_finale
    generator_backend: str = "vllm"
    generator_model: str = "qwen3.6-nvfp4"
    temperature: float = 0.0
    max_new_tokens: int = 4096
    # context of Ollama, the same for the generation and the calls of the RAG (rewriting, list mode): the longest
    # prompt measured by high-level-unisa is 17953 tokens
    num_ctx: int = 32768
    ollama_url: str = "http://localhost:11500"  # rewriting of the question and list mode, with generator_backend ollama
    vllm_url: str = "http://localhost:8001"  # the same, with generator_backend vllm
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
    # retrieval of the final configuration (ottimizzazioni_recupero.py): reranker served by vLLM (vllm_reranker of
    # servers.yaml), BM25 on a sparse matrix, 35 candidates per collection to the reranker
    reranker_backend: str = "vllm"  # vllm or locale (CrossEncoder in this process)
    reranker_url: str = "http://localhost:8002/rerank"
    reranker_fp16: bool = True  # locale only: the CrossEncoder in float16
    bm25_engine: str = "sparso"  # sparso or langchain
    rerank_candidates: int = 35
    rerank_candidates_lists: int = 35
    targeted_reasoning: bool = False  # ENABLE_RAGIONAMENTO_MIRATO, off (not supported with vLLM)


# field -> environment variable of utils/pipeline.py
ENVIRONMENT = {
    "generator_backend": "LLM_BACKEND",
    "ollama_url": "OLLAMA_BASE_URL", "num_ctx": "OLLAMA_NUM_CTX",
    "thinking": "OLLAMA_ENABLE_THINKING", "person_pipeline": "ENABLE_PERSON_PIPELINE",
    "struct_path": "ENABLE_STRUCT_PATH", "struct_attributes": "ENABLE_STRUCT_ATTRIBUTES",
    "list_mode": "ENABLE_LIST_MODE", "list_mode_top_n": "LIST_MODE_TOP_N", "name_check": "ENABLE_NAME_CHECK",
    "rerank_top_n": "RERANK_TOP_N", "bm25_k": "BM25_K", "context_order": "CONTEXT_ORDER",
    "domain_guardrail_mode": "DOMAIN_GUARDRAIL_MODE", "benchmark_date": "BENCHMARK_DATE",
    "reranker_backend": "RERANKER_BACKEND", "reranker_url": "RERANKER_URL", "reranker_fp16": "RERANKER_FP16",
    "bm25_engine": "BM25_MOTORE", "rerank_candidates": "RERANK_CANDIDATI",
    "rerank_candidates_lists": "RERANK_CANDIDATI_ELENCHI", "targeted_reasoning": "ENABLE_RAGIONAMENTO_MIRATO",
}


def load(argv=None) -> HighLevelUnisaConfig:
    return config.load(HighLevelUnisaConfig, "rag_high_level_unisa", "RAG_HIGH_LEVEL_UNISA", argv,
                       description="RAG server of the University of Salerno (Ateneo)")

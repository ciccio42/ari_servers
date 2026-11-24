"""Configuration of the orchestrator: section `orchestrator` of config/servers.yaml, ORCHESTRATOR_<FIELD>, --<field>."""
from dataclasses import dataclass

from ari_servers import config


@dataclass
class OrchestratorConfig:
    host: str = "0.0.0.0"
    port: int = 8040
    llm_url: str = "http://localhost:8010"  # the LLM server: checked by /status
    ollama_url: str = "http://localhost:11500"
    keep_alive: str = "2h"  # the model loaded stays in memory between the requests
    rag_timeout: float = 900.0  # seconds for a RAG server to answer (first start: download of its models)
    model_timeout: float = 600.0  # seconds to load a model in Ollama
    # pair loaded at the start of the orchestrator: "<llm>" and/or "<rag>" as in POST /load (empty: none)
    default_llm: str = ""
    default_rag: str = ""


def load(argv=None) -> OrchestratorConfig:
    return config.load(OrchestratorConfig, "orchestrator", "ORCHESTRATOR", argv,
                       description="orchestrator of the models: one pair LLM + RAG at a time")

"""HighLevelUnisaRag: the RAG of the University of Salerno, DiemRAG of high-level-unisa (versione_finale/pipeline.py)
behind the interface of the RAG servers (ari_servers/rag.py).

retrieve() runs the steps of main_chain of DiemRAG up to the context, without the generation:
  1. domain filter: greetings and (mode llm) out-of-domain questions answered without the model;
  2. person path: the card of the registry of people (anagrafe); fast path when the question asks a field of it;
  3. structure path: areas, offices, districts from the registry of structures; fast path for lists and counts;
  4. otherwise: rewriting of the question with the history (Ollama, only when there is a history), hybrid search
     on the four Chroma collections (BGE-M3 + BM25), list mode (three searches written by the model), reranking,
     the cards of person and structure at the head of the context;
  5. no context: the fixed answer of DiemRAG, without the model.
The LLM server generates with answer_prompt of DiemRAG (system prompt here, history, context and question) and
sends the answer back to postprocess(): the check of the names against the context (verifica_nomi).
"""
import logging
import os

from ari_servers.config import workspace_path
from ari_servers.rag import Chunk, Generator, Rag, Retrieval

from .config import ENVIRONMENT, HighLevelUnisaConfig

log = logging.getLogger("rag_high_level_unisa")


class HighLevelUnisaRag(Rag):
    name = "high_level_unisa"

    def __init__(self, cfg: HighLevelUnisaConfig):
        super().__init__(cfg)
        kb = workspace_path(cfg.kb_dir)
        # utils/pipeline.py reads its paths and switches from the environment when it is imported (in load())
        os.environ.update({
            "LLM_BACKEND": "ollama",
            "VECTORSTORE_DIR": str(kb / cfg.vectorstore),
            "ANAGRAFE_PATH": str(kb / "anagrafe.json"),
            "REGISTRO_STRUTTURE_PATH": str(kb / "registro_strutture.json"),
            "POSIZIONI_PATH": str(kb / "posizioni_parent.json"),
            "BM25_CACHE_DIR": str(kb / cfg.bm25_cache_dir) if cfg.bm25_cache_dir else "",
        })
        for name, variable in ENVIRONMENT.items():
            value = getattr(cfg, name)
            os.environ[variable] = str(value).lower() if isinstance(value, bool) else str(value)
        self.pipeline = None  # module utils.pipeline
        self.engine = None  # DiemRAG

    # --- Rag

    def load(self) -> None:
        if not workspace_path(self.cfg.kb_dir).joinpath(self.cfg.vectorstore).is_dir():
            raise SystemExit(f"index not found: {self.cfg.kb_dir}/{self.cfg.vectorstore} (kb_dir of rag_high_level_unisa)")
        from .utils import pipeline

        self.pipeline = pipeline
        self.engine = pipeline.DiemRAG()

    def retrieve(self, question: str, history=None) -> Retrieval:
        from langchain_core.messages import AIMessage, HumanMessage

        e, p = self.engine, self.pipeline
        chat_history = [AIMessage(m.get("content", "")) if m.get("role") == "assistant" else HumanMessage(m.get("content", ""))
                        for m in history or [] if m.get("role") in ("user", "assistant")]
        x = {"question": question, "chat_history": chat_history}

        status = e._domain_status(x).upper()
        if "FUORI_DOMINIO" in status:
            return Retrieval(chunks=[], answer=e.OUT_OF_DOMAIN_ANSWER, info={"path": "out_of_domain"})
        if "CONVENEVOLI" in status:
            return Retrieval(chunks=[], answer=e.GREETING_ANSWER, info={"path": "greeting"})

        persona = e._cerca_persona(question)
        rewritten = ""
        if persona and persona["veloce"]:
            path, context, docs = "person", persona["context"], persona["docs"]
        else:
            struttura = (None if persona else e._cerca_struttura(question)) if p.ENABLE_STRUCT_PATH else None
            if struttura and struttura["veloce"]:
                path, context, docs = "structure", struttura["context"], struttura["docs"]
            else:
                rewritten = e._rewrite_if_history(x)
                routing = e._router_chain.invoke({**x, "optimized_question": rewritten})
                selected = e._select_collections(routing, list(e.retrievers.keys()))
                data = e._retrieve_and_format(rewritten, selected_collections=selected)
                path = "retrieval" + ("+person" if persona else "") + ("+structure" if struttura else "")
                context = e._con_struttura(struttura, e._con_scheda(persona, data["context"]))
                docs = ((struttura["docs"] if struttura else [])
                        + (persona["docs"] if (p.PERSON_CONTEXT_BOOST and persona) else []) + data["docs"])

        chunks = [Chunk(text=d.page_content,
                        source=" | ".join(s for s in (d.metadata.get("title"), d.metadata.get("source_url")) if s),
                        score=round(float(d.metadata.get("relevance_score") or 0.0), 4)) for d in docs]
        info = {"path": path, "rewritten": rewritten if rewritten != question else "",
                "sources": [{"source": c.source, "score": c.score} for c in chunks]}
        if not context.strip():
            return Retrieval(chunks=chunks, answer=e.NO_CONTEXT_FALLBACK_ANSWER, info=info)
        system, *_, user = e._answer_prompt.format_messages(chat_history=[], context=context, question=question,
                                                            today=p.data_odierna())
        return Retrieval(chunks=chunks, system=system.content, user=user.content, info=info,
                         postprocess=p.ENABLE_NAME_CHECK, state=context)

    def postprocess(self, answer: str, state) -> tuple:
        """The names of the answer checked against the context of its retrieval (DiemRAG._verifica_nomi_risposta)."""
        if not (state or "").strip():
            return answer, {}
        corrected, replaced = self.pipeline.verifica_nomi(answer, state, self.engine.parole_comuni)
        if replaced:
            log.info("[NOMI] %s", "; ".join("%s -> %s" % r for r in replaced))
        return corrected, {"names_replaced": [list(r) for r in replaced]}

    def generator(self) -> Generator:
        # think off and the num_ctx of the calls of the RAG: a different value makes Ollama load the model again
        return Generator(self.cfg.generator_backend, self.cfg.generator_model, self.cfg.temperature,
                         self.cfg.max_new_tokens, {"num_ctx": self.cfg.num_ctx, "think": self.cfg.thinking})

    def describe(self) -> dict:
        e = self.engine
        return {**super().describe(), "collections": sorted(e.retrievers) if e else [],
                "people": len(e.anagrafe.persone) if e and e.anagrafe else 0,
                "structures": bool(e and e.registro_strutture), "name_check": self.cfg.name_check}

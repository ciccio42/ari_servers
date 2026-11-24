"""DepartmentRag: the RAG of the DIEM department, DiemRAG of DIEM_RAG (diem_RAG_routered.py) behind the interface of the
RAG servers (ari_servers/rag.py).

retrieve() runs the steps of main_chain of DiemRAG up to the context, without the generation:
  1. domain filter (the model): greetings and out-of-domain questions answered without retrieval;
  2. rewriting of the question (the model, with the history of the conversation);
  3. router (the model): the Chroma collections to search among the 13 of the department;
  4. dense search (BGE-M3, 20 chunks per collection) and reranking (bge-reranker-v2-m3, the best 7).
The LLM server generates with answer_prompt of DiemRAG: system prompt here, history, context and question.
"""
import logging
import os

from ari_servers.config import workspace_path
from ari_servers.rag import Chunk, Generator, Rag, Retrieval

from .config import DepartmentConfig

log = logging.getLogger("rag_department")


class DepartmentRag(Rag):
    name = "department"

    def __init__(self, cfg: DepartmentConfig):
        super().__init__(cfg)
        # utils/pipeline.py reads index and model from the environment when it is imported (in load())
        os.environ.update({
            "VECTORSTORE_DIR": str(workspace_path(cfg.kb_dir) / cfg.vectorstore),
            "OLLAMA_MODEL_ID": cfg.generator_model,
            "OLLAMA_BASE_URL": cfg.ollama_url,
            "OLLAMA_NUM_CTX": str(cfg.num_ctx),
        })
        self.pipeline = None  # module utils.pipeline
        self.engine = None  # DiemRAG

    def load(self) -> None:
        if not (workspace_path(self.cfg.kb_dir) / self.cfg.vectorstore).is_dir():
            raise SystemExit(f"index not found: {self.cfg.kb_dir}/{self.cfg.vectorstore} (kb_dir of rag_department)")
        from .utils import pipeline

        self.pipeline = pipeline
        self.engine = pipeline.DiemRAG()

    def retrieve(self, question: str, history=None) -> Retrieval:
        from langchain_core.messages import AIMessage, HumanMessage

        e = self.engine
        chat_history = [] if not self.cfg.use_history else [
            AIMessage(m.get("content", "")) if m.get("role") == "assistant" else HumanMessage(m.get("content", ""))
            for m in history or [] if m.get("role") in ("user", "assistant")]
        x = {"question": question, "chat_history": chat_history}

        status = e._domain_guardrail_chain.invoke(x).strip()
        log.info("[GUARDRAIL] %s", status)
        if "FUORI_DOMINIO" in status.upper():
            return Retrieval(chunks=[], answer=e.OUT_OF_DOMAIN_ANSWER, info={"path": "out_of_domain"})
        if "CONVENEVOLI" in status.upper():
            return Retrieval(chunks=[], answer=e.GREETING_ANSWER, info={"path": "greeting"})

        rewritten = e._rewrite_chain.invoke(x)
        routing = e._router_chain.invoke({**x, "optimized_question": rewritten})
        selected = e._select_collections(routing, list(e.retrievers.keys()))
        data = e._retrieve_and_format(rewritten, selected_collections=selected)
        docs = data["docs"]
        chunks = [Chunk(text=d.page_content,
                        source=" | ".join(s for s in (d.metadata.get("title"), d.metadata.get("source_url")) if s),
                        score=round(float(d.metadata.get("relevance_score") or 0.0), 4)) for d in docs]
        info = {"path": "retrieval", "rewritten": rewritten, "collections": selected,
                "sources": [{"source": c.source, "score": c.score} for c in chunks]}
        # as the original, also an empty context goes to the model (its prompt says not to invent)
        system, *_, user = e._answer_prompt.format_messages(
            chat_history=[], context=data["context"], question=question,
            today=self.pipeline.datetime.now().strftime("%d/%m/%Y"))
        return Retrieval(chunks=chunks, system=system.content, user=user.content, info=info)

    def generator(self) -> Generator:
        # the num_ctx of the calls of the RAG: a different value makes Ollama load the model again
        return Generator(self.cfg.generator_backend, self.cfg.generator_model, self.cfg.temperature,
                         self.cfg.max_new_tokens, {"num_ctx": self.cfg.num_ctx})

    def describe(self) -> dict:
        e = self.engine
        return {**super().describe(), "collections": sorted(e.retrievers) if e else [],
                "use_history": self.cfg.use_history}

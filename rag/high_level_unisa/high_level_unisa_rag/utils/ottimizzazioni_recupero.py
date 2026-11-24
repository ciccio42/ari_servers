# Copied unchanged from test_baseline/ottimizzazioni_recupero.py of the RAG repository high-level-unisa (ari_servers: rag/high_level_unisa), the module of the final configuration (doc/resoconto_attivita_5.pdf, test_baseline/sistema.sh).
# -*- coding: utf-8 -*-
"""Varianti piu' leggere del recupero (08/10/2026), tutte SPENTE per difetto: senza variabili d'ambiente la pipeline
si comporta esattamente come prima (stesso CrossEncoder, stesso BM25 di LangChain).

Perche': con vLLM il modello di generazione non e' piu' il collo di bottiglia con piu' utenti; lo sono il reranking
(8,1 volte piu' lento a 8 utenti: tutto in fila sulla GPU condivisa) e il retrieval (13,8 volte: BM25 in Python puro,
vincolato dal GIL). Misure nel diario dei test, 08/10/2026.

Interruttori (variabili d'ambiente, lette all'avvio della pipeline):

  RERANKER_FP16=true          il CrossEncoder in mezza precisione (float16): circa meta' del calcolo.
  RERANKER_MAX_LEN=512        coppie domanda+frammento tagliate a N token (0 = limite del modello, 8.192).
  RERANKER_BACKEND=vllm       il reranker non gira nel processo della pipeline ma in un server vLLM a parte
                              (avvia_reranker_vllm.sh): le richieste di piu' utenti vengono riordinate insieme dal
                              suo scheduler. RERANKER_URL (default http://127.0.0.1:8002/rerank) e RERANKER_VLLM_MODEL
                              (default bge-reranker-v2-m3). RERANKER_MAX_LEN vale anche qui (taglio lato server).
  BM25_MOTORE=sparso         BM25 calcolato con una matrice sparsa (scipy) invece che con rank_bm25, che a ogni
                              ricerca scorre in Python puro una lista lunga quanto la collezione per ogni parola
                              della domanda (e con piu' utenti si contende il GIL). Stessa formula di BM25Okapi di
                              rank_bm25 (k1=1,5, b=0,75, epsilon=0,25 per le idf negative), stessa tokenizzazione e
                              stesso k della pipeline: i punteggi sono gli stessi, quindi anche i risultati (a parte
                              l'ordine fra documenti a pari punteggio). Nessun pacchetto nuovo: numpy e scipy ci sono.
  (nella pipeline) VECTOR_K   frammenti del ramo vettoriale per collezione (default 20); con BM25_K (default 30)
                              decide quanti candidati arrivano al reranker (oggi circa 185 a domanda).
"""
import json
import logging
import os
import urllib.request
from typing import Callable, List

logger = logging.getLogger(__name__)

RERANKER_MODELLO = "BAAI/bge-reranker-v2-m3"
RERANKER_FP16 = os.getenv("RERANKER_FP16", "false").lower() in ("1", "true", "yes")
RERANKER_MAX_LEN = int(os.getenv("RERANKER_MAX_LEN", "0") or 0)
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "locale").strip().lower()
RERANKER_URL = os.getenv("RERANKER_URL", "http://127.0.0.1:8002/rerank")
RERANKER_VLLM_MODEL = os.getenv("RERANKER_VLLM_MODEL", "bge-reranker-v2-m3")
BM25_MOTORE = os.getenv("BM25_MOTORE", "langchain").strip().lower()   # "langchain" o "sparso"


def descrizione() -> str:
    """una riga per il log: quale variante e' accesa"""
    if RERANKER_BACKEND == "vllm":
        r = f"reranker vLLM ({RERANKER_URL}, max_len {RERANKER_MAX_LEN or 'modello'})"
    else:
        r = f"reranker locale ({'float16' if RERANKER_FP16 else 'float32'}, max_len {RERANKER_MAX_LEN or 'modello'})"
    return f"{r}, BM25 {BM25_MOTORE}"


# ------------------------------------------------------------------ reranker
class RerankerVLLM:
    """Stessa interfaccia del CrossEncoder usata dalla pipeline (predict(coppie) -> punteggi), ma il calcolo lo fa un
    server vLLM con l'API /rerank. Le coppie arrivano dalla pipeline tutte con la stessa domanda: una richiesta sola.
    I punteggi sono quelli del modello (per bge-reranker-v2-m3 fra 0 e 1, come il CrossEncoder con la sigmoide)."""

    def __init__(self, url: str = RERANKER_URL, modello: str = RERANKER_VLLM_MODEL, max_len: int = RERANKER_MAX_LEN):
        self.url, self.modello, self.max_len = url, modello, max_len

    def predict(self, coppie, **_):
        if not coppie:
            return []
        domanda = coppie[0][0]
        if any(q != domanda for q, _ in coppie):
            # non succede nella pipeline; per sicurezza, una richiesta per domanda
            return [p for q, d in coppie for p in self.predict([[q, d]])]
        corpo = {"model": self.modello, "query": domanda, "documents": [d for _, d in coppie]}
        if self.max_len:
            corpo["truncate_prompt_tokens"] = self.max_len
        req = urllib.request.Request(self.url, data=json.dumps(corpo).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            risposta = json.loads(r.read())
        punteggi = [0.0] * len(coppie)
        for voce in risposta.get("results", []):
            punteggi[voce["index"]] = float(voce["relevance_score"])
        return punteggi


def crea_reranker():
    """il reranker della pipeline secondo gli interruttori; senza interruttori, CrossEncoder(RERANKER_MODELLO) come
    prima"""
    if RERANKER_BACKEND == "vllm":
        return RerankerVLLM()
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANKER_MODELLO, max_length=RERANKER_MAX_LEN) if RERANKER_MAX_LEN else CrossEncoder(RERANKER_MODELLO)
    if RERANKER_FP16:
        ce.model.half()
    return ce


# ------------------------------------------------------------------ BM25
class BM25Sparso:
    """BM25Okapi (stessa formula di rank_bm25) su matrice sparsa documenti x parole, con i pesi gia' calcolati:
    il punteggio di una domanda e' un prodotto matrice-vettore (in C) invece di un ciclo Python sui documenti."""

    def __init__(self, corpus_tokenizzato, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        import numpy as np
        from scipy import sparse
        self.np = np
        n_doc = len(corpus_tokenizzato)
        vocab, righe, colonne, tf = {}, [], [], []
        lunghezze = np.zeros(n_doc, dtype=np.float64)
        df = {}
        for i, parole in enumerate(corpus_tokenizzato):
            lunghezze[i] = len(parole)
            conta = {}
            for w in parole:
                conta[w] = conta.get(w, 0) + 1
            for w, c in conta.items():
                j = vocab.setdefault(w, len(vocab))
                righe.append(i); colonne.append(j); tf.append(c)
                df[j] = df.get(j, 0) + 1
        self.vocab = vocab
        avgdl = lunghezze.sum() / n_doc if n_doc else 0.0
        # idf come in rank_bm25.BM25Okapi._calc_idf: log(N - n + 0,5) - log(n + 0,5); le negative diventano
        # epsilon * idf media (media calcolata su tutte le parole, negative comprese)
        n_par = np.zeros(len(vocab), dtype=np.float64)
        for j, n in df.items():
            n_par[j] = n
        idf = np.log(n_doc - n_par + 0.5) - np.log(n_par + 0.5)
        eps = epsilon * (idf.sum() / len(idf)) if len(idf) else 0.0
        idf[idf < 0] = eps
        righe = np.asarray(righe, dtype=np.int64); colonne = np.asarray(colonne, dtype=np.int64)
        tf = np.asarray(tf, dtype=np.float64)
        dl = lunghezze[righe]
        pesi = idf[colonne] * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)))
        self.matrice = sparse.csr_matrix((pesi, (righe, colonne)), shape=(n_doc, len(vocab)))
        self.n_doc = n_doc

    def punteggi(self, parole_domanda):
        """come BM25Okapi.get_scores: ogni parola della domanda conta tante volte quante compare; quelle che non sono
        nella collezione valgono 0"""
        np = self.np
        q = np.zeros(len(self.vocab), dtype=np.float64)
        for w in parole_domanda:
            j = self.vocab.get(w)
            if j is not None:
                q[j] += 1
        return self.matrice @ q

    def primi(self, parole_domanda, n):
        """come BM25Okapi.get_top_n: i primi n per punteggio, anche a punteggio zero"""
        return self.np.argsort(self.punteggi(parole_domanda))[::-1][:n]


def crea_bm25(documenti, tokenizzatore: Callable[[str], List[str]], k: int):
    """retriever LangChain da mettere nell'EnsembleRetriever al posto di BM25Retriever.from_documents(...)"""
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    tok = tokenizzatore or (lambda t: t.split())     # BM25Retriever senza preprocess_func usa text.split()

    class BM25SparsoRetriever(BaseRetriever):
        docs: list
        indice: object
        tok: Callable
        k: int = 30

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            return [self.docs[int(i)] for i in self.indice.primi(self.tok(query), self.k)]

    return BM25SparsoRetriever(docs=list(documenti), indice=BM25Sparso([tok(d.page_content) for d in documenti]),
                               tok=tok, k=k)

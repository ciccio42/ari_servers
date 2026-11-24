"""Queries of the Qdrant index of the department_office RAG.

From rag_pipeline/index.py of the RAG repository (rag_unisa): only the query side. The index is built there
(python -m rag_pipeline.index build) and copied into rag_knowledge_base/department_office/qdrant.
"""


from __future__ import annotations

import uuid

NOME_COLLEZIONE = "delibere"
SPAZIO = uuid.UUID("6f4a1e2c-0000-4000-8000-000000000000")

# Campi su cui si filtra: senza un indice dedicato Qdrant li scandisce.
CAMPI_FILTRO = {
    "kind": "keyword",
    "doc_type": "keyword",
    "chunk_type": "keyword",
    "source_path": "keyword",
    "doc_date": "keyword",
    "year": "integer",
}


def _id(*parti: str) -> str:
    return str(uuid.uuid5(SPAZIO, "|".join(parti)))


def _filtro(models, solo_chunk: bool = False, escludi: str | None = None, **campi):
    condizioni = []
    if solo_chunk:
        condizioni.append(models.FieldCondition(key="kind", match=models.MatchValue(value="chunk")))
    for k, v in campi.items():
        if v is not None:
            condizioni.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    esclusioni = [models.HasIdCondition(has_id=[escludi])] if escludi else []
    if not condizioni and not esclusioni:
        return None
    return models.Filter(must=condizioni or None, must_not=esclusioni or None)


def interroga(client, models, emb, limite: int, modo: str = "ibrido", filtro=None):
    """Una ricerca. Il filtro va DENTRO ogni ramo, non in cima.

    In una query con fusione il filtro passato al livello superiore non viene
    applicato ai rami: si otterrebbero risultati che violano il filtro, in
    silenzio. Verificato: chiedendo solo i verbali tornavano anche allegati.
    """
    denso = models.Prefetch(query=emb.dense[0], using="denso", limit=limite * 4, filter=filtro)
    sparso = models.Prefetch(
        query=models.SparseVector(indices=list(emb.sparse[0].keys()),
                                  values=list(emb.sparse[0].values())),
        using="lessicale", limit=limite * 4, filter=filtro,
    )
    if modo == "denso":
        return client.query_points(NOME_COLLEZIONE, query=emb.dense[0], using="denso",
                                   query_filter=filtro, limit=limite, with_payload=True).points
    if modo == "sparso":
        return client.query_points(
            NOME_COLLEZIONE,
            query=models.SparseVector(indices=list(emb.sparse[0].keys()),
                                      values=list(emb.sparse[0].values())),
            using="lessicale", query_filter=filtro, limit=limite, with_payload=True).points
    return client.query_points(
        NOME_COLLEZIONE, prefetch=[denso, sparso],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limite, with_payload=True).points

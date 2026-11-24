"""Cross-encoder reranking of the retrieved candidates.

Retrieval scores the question and each passage separately (one vector each), so a
section of the right document that shares the wording of the question can beat the
section that answers it. A cross-encoder reads question and passage together and
scores the pair: it is too slow for the whole index, but cheap on a few dozen
candidates.
"""
from __future__ import annotations

import sys
import time


class Riordinatore:
    """Wrapper around FlagReranker (bge-reranker-v2-m3 by default)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda",
                 fp16: bool = True, max_length: int = 512, batch_size: int = 16):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Manca FlagEmbedding. Installa con:  pip install FlagEmbedding") from exc

        print(f"[rerank] carico {model_name} su {device} (fp16={fp16})", file=sys.stderr)
        t0 = time.time()
        self.model = FlagReranker(model_name, use_fp16=fp16 and device != "cpu", devices=device,
                                  max_length=max_length, batch_size=batch_size)
        print(f"[rerank] modello pronto in {time.time() - t0:.1f}s", file=sys.stderr)

    def punteggi(self, domanda: str, testi: list[str]) -> list[float]:
        """Relevance of each text to the question (higher is better)."""
        if not testi:
            return []
        scores = self.model.compute_score([(domanda, t) for t in testi])
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]

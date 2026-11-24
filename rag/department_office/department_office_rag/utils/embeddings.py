

from __future__ import annotations

import sys
import time
from dataclasses import dataclass


@dataclass
class Embeddings:
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class BGEM3:
    """Involucro attorno a BGEM3FlagModel con lotti ordinati per lunghezza."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        fp16: bool = True,
        max_length: int = 1024,
        batch_size: int = 32,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Manca FlagEmbedding. Installa con:  pip install FlagEmbedding"
            ) from exc

        print(f"[embed] carico {model_name} su {device} (fp16={fp16})", file=sys.stderr)
        t0 = time.time()
        self.model = BGEM3FlagModel(model_name, use_fp16=fp16, devices=device)
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"[embed] modello pronto in {time.time() - t0:.1f}s", file=sys.stderr)

    def encode(self, testi: list[str], mostra_avanzamento: bool = True) -> Embeddings:
        if not testi:
            return Embeddings([], [])

        # I testi vanno alla GPU ordinati per lunghezza e tornano nell'ordine
        # originale: dentro un lotto tutte le sequenze vengono riempite fino
        # alla piu' lunga, quindi mescolare un testo di 50 token con uno di 500
        # fa pagare 500 token anche al primo.
        ordine = sorted(range(len(testi)), key=lambda i: len(testi[i]))
        ordinati = [testi[i] for i in ordine]

        t0 = time.time()
        out = self.model.encode(
            ordinati,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        durata = time.time() - t0
        if mostra_avanzamento:
            print(
                f"[embed] {len(testi)} testi in {durata:.0f}s "
                f"({len(testi) / max(durata, 0.01):.0f}/s)",
                file=sys.stderr,
            )

        dense = [None] * len(testi)
        sparse: list[dict[int, float]] = [None] * len(testi)
        for posizione, originale in enumerate(ordine):
            dense[originale] = [float(x) for x in out["dense_vecs"][posizione]]
            pesi = out["lexical_weights"][posizione]
            # I pesi arrivano con chiavi stringa e possono contenere zeri:
            # tenerli gonfierebbe l'indice invertito senza aggiungere segnale.
            sparse[originale] = {
                int(k): float(v) for k, v in pesi.items() if float(v) > 0.0
            }
        return Embeddings(dense, sparse)

    @property
    def dimensione(self) -> int:
        return 1024
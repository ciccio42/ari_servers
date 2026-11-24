# Attività per il 25/11/2026

## 1. Codice importato dai RAG: commenti in inglese e nomi sensati

Il codice dei tre RAG è stato copiato dai repository originali quasi senza modifiche: i commenti, le docstring, i
nomi di file, funzioni e variabili sono in italiano e i file hanno i nomi delle prove (`rispondi.py`,
`diem_RAG_routered_prova_elenchi_...`). Vanno riscritti in inglese, con nomi che dicano che cosa fanno, come il resto
di `ari_servers`.

| RAG | File in `utils/` | Righe |
|---|---|---|
| department_office | `rispondi.py`, `registers.py`, `index.py`, `tables.py`, `embeddings.py`, `rerank.py` | 2.291 |
| high_level_unisa | `pipeline.py`, `struttura_lookup.py`, `anagrafe_lookup.py`, `ottimizzazioni_recupero.py`, `modalita_elenco.py`, `verifica_nomi.py` | 3.427 |
| department | `pipeline.py` | 563 |

- [ ] Proposta dei nuovi nomi (file, classi, funzioni, variabili d'ambiente) da approvare **prima** di cambiare il codice,
      per esempio `rispondi.py` → `answer.py`, `Fonte` → `Source`, `recupera` → `retrieve`,
      `ottimizzazioni_recupero.py` → `retrieval_variants.py`, `anagrafe_lookup.py` → `people_registry.py`.
- [ ] Commenti e docstring in inglese, senza perdere le misure e le motivazioni scritte dagli autori.
- [ ] **Non tradurre** i prompt dati al modello (`ISTRUZIONI`, `answer_prompt`, `PLANNER_PROMPT`, prompt di riscrittura e
      router): cambierebbero le risposte. Restano in italiano, come le risposte fisse ("Nei documenti non trovo...").
- [ ] Variabili d'ambiente rinominate solo dentro `ari_servers`: aggiornare insieme `config.py`, `ENVIRONMENT`,
      `rag.py` e `config/servers.yaml`.
- [ ] Togliere il codice morto della configurazione finale (rami spenti: router, riparazione, xprovence, riassunto,
      HF pipeline), solo dove è sicuro e annotandolo nel README del RAG.
- [ ] Aggiornare le tabelle "Here ↔ original" dei README dei RAG con i nuovi nomi e i file d'origine.
- [ ] Verifica: per ogni RAG, `/retrieve` sulle stesse domande prima e dopo deve dare **lo stesso contesto** (stesso
      hash di `user`, stessi chunk); poi una prova completa con `/chat` tramite l'orchestratore.

## Da verificare o chiudere dalla giornata precedente

- [x] **high_level_unisa sulla configurazione del resoconto** (vLLM 0.31 con MTP, reranker vLLM, pipeline `_cronologia`):
      caricata con l'orchestratore (436 s la prima volta: lettura dei pesi e compilazione), 1,3–3,8 s per risposta
      (7,2 s su una domanda di elenco), "E la sua matricola?" ora passa dal percorso persona.
- [ ] Misurare con più domande (giro F o banchi) la media con un utente, da confrontare con i 2,92 s del resoconto, e
      il tempo di un cambio di coppia a caldo (cache di compilazione già pronte).
- [ ] Cache su disco di BM25 e parole comuni: la pipeline finale usa il BM25 su matrice sparsa e la cache della
      versione precedente non c'è più. Misurare l'avvio e decidere se rifarla (`ari_servers/rag/high_level_unisa`).
- [ ] ef_search: il resoconto dice 400, l'indice in uso ha la configurazione predefinita (100). Chiarire con gli autori.
- [ ] Protezione della memoria di `sistema.sh` (sotto 15 GB disponibili spegne motore e reranker) non portata
      nell'orchestratore: decidere se aggiungerla.
- [ ] L'orchestratore perde la coppia attiva quando si riavvia: ricostruirla all'avvio da ciò che è acceso.
- [ ] Immagine `vllm/vllm-openai:v0.23.0` scaricata e non più usata: cancellarla (per ID, solo dopo conferma).
- [x] Documentazione di vLLM (motore e reranker) e dei servizi dei RAG: README principale, `rag/README.md`,
      `rag/high_level_unisa/README.md` (riscritto sulla configurazione del resoconto), `llm/README.md`,
      `orchestrator/README.md`, `docs/integration.md`.
- [ ] Commit delle modifiche di `ari_servers` (nessun commit fatto finora).

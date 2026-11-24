# RAG high_level_unisa

University of Salerno, at the level of the whole Ateneo: governing bodies, rector and delegates, departments and
directors, central administration (areas, offices, people), statute and regulations, history, quality and
evaluation. Built by the repository **high-level-unisa** (M.S. Grimaldi, optimized by Iannone; `tmp/high-level-unisa`
on the Spark); here only the side that answers, with its **final configuration**: `doc/resoconto_attivita_5.pdf`
(20/11/2026) and `test_baseline/sistema.sh`.

| | |
|---|---|
| Server | `rag_high_level_unisa`, port 8031, section `rag_high_level_unisa` of `config/servers.yaml` |
| Knowledge base | `rag_knowledge_base/high_level_unisa`: `diem_vectorstore/` (Chroma, 4 collections: ateneo_generale, percorsi_di_studio, statuto_e_normativa, terza_missione; 175659 chunks), `anagrafe.json` (people), `registro_strutture.json` (areas, offices, districts), `posizioni_parent.json` (only for the neighbourhood expansion, off) |
| Retrieval | BGE-M3 (`BAAI/bge-m3`, 20 chunks per collection) + BM25 on a sparse matrix (30), 35 candidates per collection to the reranker `bge-reranker-v2-m3` **served by vLLM** (`vllm_reranker`), the best 10 (20 in list mode); conda environment `rag_high_level_unisa` |
| Model calls of the RAG | rewriting of the question (only with a history) and list mode (three searches for the list questions), on the vLLM engine |
| Answers | LLM server, backend `vllm`: `qwen3.6-nvfp4` (Qwen3.6-35B-A3B in NVFP4) on **vLLM 0.31 with MTP**, temperature 0, 4096 tokens, reasoning off |
| After the answer | check of the names against the context (`verifica_nomi`), `POST /postprocess` called by the LLM server |
| Services | `/health` declares `"services": ["vllm_reranker"]`: the orchestrator starts the reranker with the engine |

## Servers of the pair
Loaded together by the orchestrator (`POST :8040/load {"rag": "high_level_unisa"}`), with
[vllm/docker/run.bash](../../vllm/docker/run.bash):

| Container | Port | What | Settings (report, `sistema.sh`) |
|---|---|---|---|
| `rag_high_level_unisa` | 8031 | retrieval (this RAG) | BGE-M3 on the GPU, BM25 in memory |
| `vllm_server` (engine) | 8001 | `nvidia/Qwen3.6-35B-A3B-NVFP4` served as `qwen3.6-nvfp4`, OpenAI API | `vllm/vllm-openai:v0.31.0`, MTP with 3 proposed tokens, `--max-num-seqs 16`, `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.40` (about 49 GB), prefix caching on, `--max-model-len 131072`, modelopt / marlin / flashinfer / KV cache fp8 |
| `vllm_reranker` | 8002 | `BAAI/bge-reranker-v2-m3`, `/rerank` | `vllm/vllm-openai:v0.31.0`, `--runner pooling`, `--kv-cache-memory-bytes 536870912` (0.5 GB fixed, about 6.2 GB in all), `--max-model-len 8192`, `--max-num-seqs 256` |

Models in the Hugging Face cache of `ari_servers`: the engine in `models/vllm` (`bash vllm/docker/run.bash download`,
revision fixed in servers.yaml), the reranker in `models/rag/.hf_cache` (the one of the RAG servers). Compile caches
of vLLM: Docker volume `ari-vllm-cache-031`.

Measured on the Spark (2026-11-24): first load of the pair 436 s (the engine reads the weights and compiles), the RAG
alone 43 s; answers in 1.3-3.8 s (7.2 s for a list question), against 5.6-40 s of the original on Ollama on the same
questions of the run F (the report: 2.92 s on average over 1030 questions, 9.18 s on Ollama). Memory: about 43 GB
still available with the pair loaded.

## Flow of a question
```
POST /chat {"rag": "high_level_unisa"} ─► llm_server ─POST /retrieve {question, history}─► rag_high_level_unisa
                                                        greetings (rules) ───────────────► fixed answer
                                                        rewriting (history) ─────────────► vllm_server :8001
                                                        person path (anagrafe), structure path (registry), on the
                                                        rewritten question ─────────────► card as context (fast path)
                                                        hybrid search on the 4 collections, list mode (vllm_server),
                                                        reranking ──────────────────────► vllm_reranker :8002
                                                        cards at the head of the context
                                            ◄─ system prompt, context + question, generator, retrieval_id
                         ─► vllm_server qwen3.6-nvfp4 (system, history, context + question)
                         ─POST /postprocess {retrieval_id, answer}─► names checked against the context
                         ◄─ text
```
The rewriting comes **before** the person and structure paths (the fix of the report): "E la sua matricola?" after a
question on a person is rewritten with the name and reaches the card of the registry.

## From high-level-unisa
| Here | high-level-unisa |
|---|---|
| `high_level_unisa_rag/rag.py`, `HighLevelUnisaRag` | `DiemRAG.retrieve_and_generate` and `main_chain` of the pipeline, step by step up to the context |
| `utils/pipeline.py` | `test_baseline/diem_RAG_routered_prova_elenchi_ottimizzazioni_reranker_cronologia.py`, the pipeline of the chatbot of `sistema.sh`: relative imports, no `.env`, Ollama clients with `base_url` and `num_ctx` (backend `ollama` only), the pieces of the LCEL chain kept on the instance; the rest unchanged |
| `utils/ottimizzazioni_recupero.py` | `test_baseline/ottimizzazioni_recupero.py`: reranker on vLLM, BM25 on a sparse matrix; unchanged |
| `utils/anagrafe_lookup.py`, `struttura_lookup.py`, `modalita_elenco.py`, `verifica_nomi.py` | the modules of the project folder used by the pipeline; unchanged |
| settings of `PIPELINE_ENV` (`sistema.sh`) and `CONFIGURAZIONE` (`telegram_bot_vllm.py`) | fields of [config.py](high_level_unisa_rag/config.py), set as environment variables of the pipeline (`ENVIRONMENT`) |
| `avvia_vllm_esperimenti.sh`, the container `vllm-reranker-031-kv05g` (`coda_rr_0511.sh`) | [vllm/docker/run.bash](../../vllm/docker/run.bash) `engine` / `reranker`, sections `vllm` and `vllm_reranker` of servers.yaml |
| generation (`answer_prompt \| llm`) and history (`RunnableWithMessageHistory`) | LLM server: the same prompt (system, history, `INFORMAZIONI A TUA DISPOSIZIONE`), the history of the conversation of the LLM server |
| `telegram_bot_vllm.py`, `sistema.sh`, tests, calibrations, judge, `costruzione/` | not ported: the knowledge base is built there and copied in `rag_knowledge_base/high_level_unisa` |

Differences from `sistema.sh`:
- the start of the engine needs `gpu_memory_utilization x total + 26 GB` available (as `sistema.sh`); its guard that
  stops engine and reranker when the available memory falls under 15 GB with the chatbot on is **not ported**;
- the report gives `ef_search 400`, but the index in use (also by its chatbot) has the default HNSW configuration
  (100): kept as it is, to be checked with the authors.

The other switches of the pipeline keep the final configuration: router, guardrail with the model, repair, page and
neighbourhood expansion, chunk filter, summary, XProvence and targeted reasoning off.

## Configuration
Section `rag_high_level_unisa` of `config/servers.yaml`, env `RAG_HIGH_LEVEL_UNISA_<FIELD>`: `kb_dir`, `vectorstore`,
`generator_backend` (`vllm`; `ollama` with `qwen3.6:35b-a3b`, the model of `versione_finale`), `generator_model`,
`temperature`, `max_new_tokens`, `vllm_url`, `ollama_url`, `num_ctx` (Ollama only), `thinking`, `person_pipeline`,
`struct_path`, `struct_attributes`, `list_mode`, `list_mode_top_n`, `name_check`, `rerank_top_n` (10), `bm25_k` (30),
`context_order` (coda), `domain_guardrail_mode` (permissivo), `benchmark_date` (empty: today in the prompt),
`reranker_backend` (`vllm`; `locale`: CrossEncoder in the RAG, no service), `reranker_url`, `reranker_fp16`,
`bm25_engine` (`sparso`), `rerank_candidates` (35), `rerank_candidates_lists` (35), `targeted_reasoning`.
Engine and reranker: sections `vllm` and `vllm_reranker`. See [config.py](high_level_unisa_rag/config.py).

Without Docker, in the conda environment: `PYTHONPATH=$PWD/common:$PWD/rag/high_level_unisa python -m high_level_unisa_rag`.

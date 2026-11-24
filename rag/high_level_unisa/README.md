# RAG high_level_unisa

University of Salerno, at the level of the whole Ateneo: governing bodies, rector and delegates, departments and
directors, central administration (areas, offices, people), statute and regulations, history, quality and
evaluation. Built by the repository **high-level-unisa** (M.S. Grimaldi, optimized by Iannone;
`tmp/high-level-unisa` on the Spark, `versione_finale/`); here only the side that answers.

| | |
|---|---|
| Server | `rag_high_level_unisa`, port 8031, section `rag_high_level_unisa` of `config/servers.yaml` |
| Knowledge base | `rag_knowledge_base/high_level_unisa`: `diem_vectorstore/` (Chroma, 4 collections: ateneo_generale, percorsi_di_studio, statuto_e_normativa, terza_missione; 175659 chunks), `anagrafe.json` (people), `registro_strutture.json` (areas, offices, districts), `posizioni_parent.json` (only for the neighbourhood expansion, off) |
| Retrieval | BGE-M3 (`BAAI/bge-m3`) + BM25 per collection, reranker `BAAI/bge-reranker-v2-m3` (CrossEncoder), conda environment `rag_high_level_unisa` |
| Model calls of the RAG | rewriting of the question (only with a history) and list mode (three searches for the list questions), on Ollama |
| Answers | LLM server, backend `ollama`, `qwen3.6:35b-a3b` (the model of the evaluations of high-level-unisa), temperature 0, 4096 tokens, `num_ctx` 32768, reasoning off |
| After the answer | check of the names against the context (`verifica_nomi`), `POST /postprocess` called by the LLM server |

## Flow of a question
```
POST /chat {"rag": "high_level_unisa"} ─► llm_server ─POST /retrieve {question, history}─► rag_high_level_unisa
                                                        greetings (rules) ───────────────► fixed answer
                                                        person path (anagrafe) ──────────► card as context (fast path)
                                                        structure path (registry) ───────► card as context (fast path)
                                                        rewriting (history, Ollama), hybrid search on the 4 collections,
                                                        list mode (Ollama), reranking, cards at the head of the context
                                            ◄─ system prompt, context + question, generator, retrieval_id
                         ─► Ollama qwen3.6:35b-a3b (system, history, context + question)
                         ─POST /postprocess {retrieval_id, answer}─► names checked against the context
                         ◄─ text
```

## From high-level-unisa
| Here | high-level-unisa (`versione_finale/`) |
|---|---|
| `high_level_unisa_rag/rag.py`, `HighLevelUnisaRag` | `DiemRAG.retrieve_and_generate` and `main_chain` of `pipeline.py`, step by step up to the context |
| `utils/pipeline.py` | `pipeline.py`: relative imports, no `.env`, Ollama clients with `base_url` and `num_ctx`, the pieces of the LCEL chain kept on the instance, the cache on disk of BM25 (below); the rest unchanged |
| `utils/percorso_persona.py`, `percorso_struttura.py`, `modalita_elenco.py`, `verifica_nomi.py` | unchanged |
| generation (`answer_prompt \| llm`) and history (`RunnableWithMessageHistory`) | LLM server: the same prompt (system, history, `INFORMAZIONI A TUA DISPOSIZIONE`), the history of the conversation of the LLM server |
| `bot_telegram.py`, tests, judge, `costruzione/` (index, registries, scraping) | not ported: the knowledge base is built there and copied in `rag_knowledge_base/high_level_unisa` |

The switches of `pipeline.py` keep the final configuration (`LEGGIMI.md`): router, guardrail with the model,
repair, page and neighbourhood expansion, chunk filter, summary and XProvence off. The ones in the configuration
below become the environment variables of `pipeline.py` (`config.py`, `ENVIRONMENT`); the others can still be set
in the environment of the container.

## Start and cache of BM25
At the start the RAG loads BGE-M3 (12 s), the reranker (17 s) and, for each of the 4 collections, a BM25 retriever
over all its chunks and the lowercase words used by the check of the names. Built from Chroma they take about 20 s
(more with the disk cold: the index is 3.3 GB); the first start writes them in `rag_knowledge_base/high_level_unisa/cache/`
(4 files, 950 MB) and the next ones read them in 2.5 s. Start measured on the Spark: 57 s building the cache, 40 s
reading it, same retrieval. A file is valid while its collection has the same chunks (count, first and last id) and
the same tokenization: after a new index it is built again by itself. `bm25_cache_dir: ""` disables the cache; to
build it again, delete the folder.

## Configuration
Section `rag_high_level_unisa` of `config/servers.yaml`, env `RAG_HIGH_LEVEL_UNISA_<FIELD>`: `kb_dir`,
`vectorstore`, `generator_model`, `temperature`, `max_new_tokens`, `num_ctx`, `ollama_url`, `thinking`,
`person_pipeline`, `struct_path`, `struct_attributes`, `list_mode`, `list_mode_top_n`, `name_check`, `rerank_top_n`
(10), `bm25_k` (30), `context_order` (coda), `domain_guardrail_mode` (permissivo), `benchmark_date` (empty: today in
the prompt), `bm25_cache_dir` (`cache`, relative to `kb_dir`; empty: no cache). See [config.py](high_level_unisa_rag/config.py).

Without Docker, in the conda environment: `PYTHONPATH=$PWD/common:$PWD/rag/high_level_unisa python -m high_level_unisa_rag`.

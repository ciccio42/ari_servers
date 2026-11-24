# RAG department

The DIEM department of the University of Salerno: courses (bachelor, master, PhD), people and their courses, research
areas and projects, calls, Erasmus, bodies and commissions. Built by the RAG of the course Generative AI, local
version (**DIEM_RAG**, `tmp/deparment` on the Spark); here only the side that answers.

| | |
|---|---|
| Server | `rag_department`, port 8032, section `rag_department` of `config/servers.yaml` |
| Knowledge base | `rag_knowledge_base/department/vectorstore` (Chroma, 13 collections: bandi, didattica, personale, ricerca, international, dipartimento and one per course; 20431 chunks) |
| Retrieval | BGE-M3 (`BAAI/bge-m3`, 20 chunks per collection), reranker `BAAI/bge-reranker-v2-m3` (CrossEncoder, the best 7), conda environment `rag_department` |
| Model calls of the RAG | domain filter, rewriting of the question, router of the collections (JSON), on Ollama |
| Answers | LLM server, backend `ollama`, `qwen3:30b-a3b-instruct-2507-fp16`: the Qwen3-30B-A3B-Instruct-2507 of the original (transformers, bf16, greedy), fp16, temperature 0, 4096 tokens, `num_ctx` 16384 |

## Flow of a question
```
POST /chat {"rag": "department"} ─► llm_server ─POST /retrieve {question, history}─► rag_department
                                                    domain filter (Ollama) ─► greetings, out of domain: fixed answer
                                                    rewriting (Ollama, history), router (Ollama): collections
                                                    dense search, reranking: the best 7 chunks
                                    ◄─ system prompt, context + question, generator
                     ─► Ollama qwen3:30b-a3b-instruct-2507-fp16 (system, history, context + question) ─► text
```

## From DIEM_RAG
| Here | DIEM_RAG |
|---|---|
| `department_rag/rag.py`, `DepartmentRag` | `DiemRAG.retrieve_and_generate` and `main_chain` of `diem_RAG_routered.py`, step by step up to the context |
| `utils/pipeline.py` | `diem_RAG_routered.py`: no `.env`, index and model from the environment, `ChatOllama` instead of `HuggingFacePipeline` (same model), the pieces of the LCEL chain kept on the instance; the rest unchanged |
| generation (`answer_prompt \| llm`) | LLM server: the same prompt (system, history, `INFORMAZIONI A TUA DISPOSIZIONE`) |
| `main.py` (bot Telegram), `crawling/` (scraping, `build_vectorstore.py`), `testing/` | not ported: the index is built there and copied in `rag_knowledge_base/department/vectorstore` |

Differences from the original:
- **model on Ollama**: the same weights (fp16 instead of bf16), on the Ollama server shared by the RAGs and loaded by
  the orchestrator, instead of a second copy in the process of the RAG;
- **history**: the original gave an empty history at every request (`get_session_history`, marked as a test to undo);
  here the rewriting and the answer get the history of the conversation (`use_history: false`: as the original).
  The domain filter, as in the original, reads only the last question: a follow-up such as "E qual è la sua email?"
  can be classified out of domain before the rewriting resolves it. Kept on purpose, to leave the logic of DIEM_RAG
  unchanged.

## Configuration
Section `rag_department` of `config/servers.yaml`, env `RAG_DEPARTMENT_<FIELD>`: `kb_dir`, `vectorstore`,
`generator_model`, `temperature`, `max_new_tokens`, `num_ctx`, `ollama_url`, `use_history`. See
[config.py](department_rag/config.py).

Without Docker, in the conda environment: `PYTHONPATH=$PWD/common:$PWD/rag/department python -m department_rag`.

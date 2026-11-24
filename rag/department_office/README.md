# RAG department_office

Resolutions of the Consiglio Didattico di Ingegneria Informatica (UNISA): 394 documents 2021-2026 (minutes,
calls, attendance sheets, course sheets, SUA-CdS, regulations, decrees). Built by the repository **rag_unisa**
(M.S. D'Acunto, `tmp/deparment_office` on the Spark); here only the side that answers.

| | |
|---|---|
| Server | `rag_department_office`, port 8030, section `rag_department_office` of `config/servers.yaml` |
| Knowledge base | `rag_knowledge_base/department_office`: `qdrant/` (collection `delibere`, 53487 points: chunks and generated questions, dense + sparse BGE-M3), `registers.json` (attendance of 59 meetings), `chunks/` (`chunks_v7_enriched`, not read by the server) |
| Retrieval | BGE-M3 (`BAAI/bge-m3`), reranker `BAAI/bge-reranker-v2-m3`, conda environment `rag_department_office` |
| Answers | LLM server, backend `ollama`, `qwen3.6:35b-a3b-q8_0` (the model of the evaluations of rag_unisa), `num_ctx` 16384, temperature 0.1, 800 tokens |

## From rag_unisa
| Here | rag_unisa (`rag_pipeline/`) |
|---|---|
| `department_office_rag/rag.py`, `DepartmentOfficeRag` | `Assistente` of `rispondi.py`: `cerca`, `query_registers`, `_ultime_versioni` |
| `utils/rispondi.py` | `rispondi.py` without `Assistente` and the CLI; the date of the prompt is computed at every request |
| `utils/index.py` | `index.py`, only the queries (`interroga`, `_filtro`, `_id`) |
| `utils/registers.py` | `registers.py` without the diagnosis and the CLI (the build stays: `build_registers`) |
| `utils/embeddings.py`, `utils/rerank.py`, `utils/tables.py` | unchanged |
| generation (`genera`, streaming to Ollama) | LLM server, backend `ollama`, not streamed: the prompt (`ISTRUZIONI`) and the request come from `POST /retrieve` |

The planner of the registers (questions over many meetings: attendance, number of meetings, mentions) asks Ollama
with tool calls from the RAG server, during the retrieval, as in rag_unisa. The citations of the registers
(`[Registro presenze (calcolato)]`) are removed from the answer by the LLM server (`drop_patterns`).

Not ported: building the knowledge base (convert, chunk, enrich, index build, registers build), the Gradio
interface, the evaluation. A new knowledge base is built in rag_unisa and copied in `rag_knowledge_base/department_office`.

## Configuration
Section `rag_department_office` of `config/servers.yaml`, env `RAG_DEPARTMENT_OFFICE_<FIELD>`: `kb_dir`,
`generator_model`, `temperature`, `max_new_tokens`, `num_ctx`, `ollama_url`, `embedding_model`, `reranker_model`
(empty: off), `top_k` (14), `candidates` (40), `mode` (ibrido, denso, sparso), `expansion` (10), `context_max_chars`
(30000), `use_registers`, `latest_versions`, `complete_sentences`. See [config.py](department_office_rag/config.py).

Without Docker, in the conda environment: `PYTHONPATH=$PWD/common:$PWD/rag/department_office python -m department_office_rag`.

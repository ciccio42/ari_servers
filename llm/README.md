# LLM (+RAG) server — skeleton

Generation of the answers of the demo, with retrieval on the documents (RAG). In place: API, configuration (model as
a parameter), execution lock, latency log. To do: the backends, ported from PepperDemo
(`ros2_ws/src/trees/trees/llm_rag_utils`).

| File | Content | State |
|---|---|---|
| `llm_server/config.py` | `LLMConfig`: section `llm` of `config/servers.yaml`, `LLM_<FIELD>`, `--<field>` | done |
| `llm_server/server.py` | `LLMService`: `GET /health`, `POST /chat` | done |
| `llm_server/backend.py` | `ChatBackend` (interface), `TransformersBackend`, `BACKENDS` | TODO: port `chatmodel.py` |
| `llm_server/rag.py` | `Retriever` (interface), `FaissRetriever`, `Chunk` | TODO: port `rag.py`, `retrieval.py` |

## API
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service", "backend", "model", "rag", "loaded"}` |
| `POST /chat` | `{"messages": [{"role", "content"}...]}` or `{"prompt": "...", "system": "..."}`; optional `rag`, `max_new_tokens`, `temperature` |
| reply | `{"text", "chunks": [{"text", "source", "score"}], "rag_s", "generate_s", "queue_s", "total_s"}` |
| errors | `400` bad request, `501` backend not implemented or models not loaded, `500` generation failed |

## Run
```bash
export PYTHONPATH=$PWD/common:$PWD/llm      # from ari_servers
python -m llm_server --skip-load            # without models: /health answers, /chat replies 501
python -m llm_server --model meta-llama/Llama-3.2-3B-Instruct --rag false
```

## Next steps
1. `TransformersBackend`: load and generate as `ChatModel` (model in `models/llm/<name>`, as `ensure_model_available`).
2. `FaissRetriever`: index of `prepare_rag.py`, BM25 + FAISS, reranker, as `Rag`.
3. System prompt: `system_prompt_file` (the prompts of `service.py`, e.g. the medical one).
4. Image (Spark) and conda environment, as for Whisper.

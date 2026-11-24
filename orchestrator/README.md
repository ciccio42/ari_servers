# Orchestrator of the models

The memory of the Spark does not hold every model at once (two qwen3.6 of the RAGs: 37 + 22 GB, plus their
embedding models and rerankers). The orchestrator is called from outside (`ari_control`, a demo, a script) and
loads **one pair LLM + RAG at a time**, unloading the previous one. Whisper, TTS, RASA and the LLM server are not
touched. After the load, the clients use the pair through the LLM server as before.

| Pair | Request | What is loaded | Then the clients call |
|---|---|---|---|
| RAG (its model) | `{"rag": "high_level_unisa"}` | container `rag_high_level_unisa`, `qwen3.6:35b-a3b` in Ollama (with the `num_ctx` of the RAG) | `POST :8010/chat {"rag": "high_level_unisa", ...}` |
| RAG | `{"rag": "department_office"}` | container `rag_department_office`, `qwen3.6:35b-a3b-q8_0` | `POST :8010/chat {"rag": "department_office", ...}` |
| RAG | `{"rag": "department"}` | container `rag_department`, `qwen3:30b-a3b-instruct-2507-fp16` (61 GB) | `POST :8010/chat {"rag": "department", ...}` |
| local LLM, no RAG | `{"llm": "ollama:qwen3.6:35b-a3b"}` | the model in Ollama, no RAG | `POST :8010/chat {"llm": "ollama:qwen3.6:35b-a3b", ...}` |
| ChatGPT | `{"llm": "openai:gpt-4o"}` | nothing (the model is at OpenAI): every RAG and local model unloaded | `POST :8010/chat {...}` (default of the LLM server) |

`llm` and `rag` together are accepted only when `llm` is the generator of the RAG (its section in servers.yaml):
the model of the answers of a RAG is the one it was evaluated with.

## API (`<ORCHESTRATOR_PORT>`, default 8040)
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service": "orchestrator", "state", "active"}` |
| `GET /status` | `{"state": "idle\|loading\|ready\|failed", "active": {"llm", "rag"}, "requested", "error", "load_s", "ollama": [models in memory], "rags": {name: {"running", "ready"}}, "llm_server"}` |
| `POST /load` | `{"llm": "<backend>:<model>", "rag": "<name>", "wait": true}` → the status. `wait: false` replies at once (`state: loading`), then `GET /status`. The same pair already in memory: immediate |
| `POST /unload` | stops every RAG container and unloads every model of Ollama |
| errors | `400` unknown RAG or backend, `409` a load is running (with `wait: false`), `502` the load failed (`error`) |

A load: stop the other RAG containers, start Ollama if needed, start the RAG container and wait for its `/health`
(`loaded`), unload the other models of Ollama, load the model with `keep_alive` (2 h) and the `num_ctx` of the RAG,
so the first question does not wait for it. One load at a time: a second `POST /load` waits for the first.

Measured on the Spark (2026-11-24): `department_office` 2-8 s, `high_level_unisa` about 45 s (BGE-M3 and reranker
about 30 s, BM25 from its cache on disk 2.5 s, model 6 s; 141 s the first time, with the disk cold and without the
cache), a local LLM alone 7 s, ChatGPT 0.6 s. Memory available: 106 GB with ChatGPT, 83 GB with qwen3.6:35b-a3b alone, 59-73 GB with a RAG, 19 GB with
both RAGs loaded together.

## Run (Spark)
From `ari_servers`:
```bash
bash orchestrator/docker/run.bash build            # once: image ari_orchestrator (Python, Docker client)
bash orchestrator/docker/run.bash start            # container orchestrator_server; also: bash run_servers.bash start orchestrator
bash orchestrator/docker/run.bash load rag=high_level_unisa
bash orchestrator/docker/run.bash load llm=ollama:qwen3.6:35b-a3b
bash orchestrator/docker/run.bash status
bash orchestrator/docker/run.bash unload
curl -X POST -H 'Content-Type: application/json' localhost:8040/load -d '{"rag": "department_office"}'
```
Section `orchestrator` of `config/servers.yaml` (env `ORCHESTRATOR_<FIELD>`): `port`, `keep_alive`, `rag_timeout`,
`model_timeout`, `default_rag` / `default_llm` (pair loaded when the orchestrator starts), `llm_url`, `ollama_url`.

The container has the socket of Docker and `ari_servers` at its path of the host: it starts and stops the RAG
containers and Ollama with their `run.bash`, which mount the same folders as from the host. It does not use the GPU
and does not take the execution lock.

| Path | Content |
|---|---|
| `orchestrator_server/server.py` | `Orchestrator` (the steps of a load), `OrchestratorService` (HTTP, on `HTTPService`) |
| `orchestrator_server/config.py` | `OrchestratorConfig` |
| `docker/` | image `ari_orchestrator`, `entrypoint.bash`, `run.bash` |

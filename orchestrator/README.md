# Orchestrator of the models

The memory of the Spark does not hold every model at once (the models of the RAGs: 49 GB on vLLM, 38 and 61 GB on
Ollama, plus their embedding models and rerankers). The orchestrator is called from outside (`ari_control`, a demo, a
script) and loads **one pair LLM + RAG at a time**, unloading the previous one. Whisper, TTS, RASA and the LLM server
are not touched. After the load, the clients use the pair through the LLM server as before.

| Pair | Request | What is loaded | Then the clients call |
|---|---|---|---|
| RAG (its model) | `{"rag": "high_level_unisa"}` | container `rag_high_level_unisa`, the vLLM engine `vllm_server` (`qwen3.6-nvfp4`) and the service `vllm_reranker` | `POST :8010/chat {"rag": "high_level_unisa", ...}` |
| RAG | `{"rag": "department_office"}` | container `rag_department_office`, `qwen3.6:35b-a3b-q8_0` in Ollama (with the `num_ctx` of the RAG) | `POST :8010/chat {"rag": "department_office", ...}` |
| RAG | `{"rag": "department"}` | container `rag_department`, `qwen3:30b-a3b-instruct-2507-fp16` in Ollama (61 GB) | `POST :8010/chat {"rag": "department", ...}` |
| local LLM, no RAG | `{"llm": "ollama:qwen3.6:35b-a3b"}` or `{"llm": "vllm:qwen3.6-nvfp4"}` | the model in Ollama, or the vLLM engine; no RAG | `POST :8010/chat {"llm": "<the same>", ...}` |
| ChatGPT | `{"llm": "openai:gpt-4o"}` | nothing (the model is at OpenAI): every RAG, local model and vLLM container stopped | `POST :8010/chat {...}` (default of the LLM server) |

`llm` and `rag` together are accepted only when `llm` is the generator of the RAG (its section in servers.yaml):
the model of the answers of a RAG is the one it was evaluated with.

## Engines and services
| Backend / service | Started with | Ready when | Stopped |
|---|---|---|---|
| `ollama:<model>` | `ollama/docker/run.bash start` (if Ollama is down), then the model loaded with `POST /api/generate` (`keep_alive`, `num_ctx` of the RAG) | the load returns | `keep_alive: 0` for the models of the other pairs |
| `vllm:<served model>` | `vllm/docker/run.bash start engine` (container `vllm_server`) | `GET :8001/v1/models` lists the model (first start: minutes, compilation) | `vllm/docker/run.bash stop engine` for every pair that does not use it |
| services of a RAG | the RAG declares them in `"services"` of its `/health`; known: `vllm_reranker` → `vllm/docker/run.bash start reranker` (container `vllm_reranker`) | `GET :8002/v1/models` answers | stopped for every pair whose RAG does not declare them |

## API (`<ORCHESTRATOR_PORT>`, default 8040)
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service": "orchestrator", "state", "active"}` |
| `GET /status` | `{"state": "idle\|loading\|ready\|failed", "active": {"llm", "rag"}, "requested", "error", "load_s", "ollama": [models in memory], "vllm": [models served] or null, "services": {name: ready}, "rags": {name: {"running", "ready"}}, "llm_server"}` |
| `POST /load` | `{"llm": "<backend>:<model>", "rag": "<name>", "wait": true}` → the status. `wait: false` replies at once (`state: loading`), then `GET /status`. The same pair already in memory: immediate |
| `POST /unload` | stops every RAG container, the vLLM engine and the services, unloads every model of Ollama |
| errors | `400` unknown RAG or backend (the backends are read from `llm/llm_server/backends`), `409` a load is running (with `wait: false`), `502` the load failed (`error`) |

A load:
1. stop the other RAG containers; without a RAG, stop the services; with an `llm` that is not `vllm:`, stop vLLM;
2. start the RAG container and wait for its `/health` (`loaded`): its `generator` gives the model of the pair, its
   `services` the other servers it needs; stop the services it does not declare;
3. stop vLLM if the generator is not on vLLM; unload the other models of Ollama;
4. load the model in Ollama (with the `num_ctx` of the RAG, so the first question does not wait for it), or start
   the vLLM engine and wait until it serves the model;
5. start the services of the RAG and wait until they answer.

One load at a time: a second `POST /load` waits for the first. The memory is freed before the new pair is loaded;
the start of the vLLM engine needs `gpu_memory_utilization x total + reserve_gb` available (section `vllm`).

Measured on the Spark (2026-11-24): `department_office` 2-8 s, `department` 43 s, `high_level_unisa` 436 s the first
time (the vLLM engine reads the weights and compiles; RAG alone 43 s, reranker 55 s), a local LLM alone 7 s, ChatGPT
0.6 s. Memory available: 106 GB with ChatGPT, 83 GB with qwen3.6:35b-a3b alone, 42-73 GB with a RAG (43 GB with
`high_level_unisa` on vLLM), 19 GB with two RAGs loaded together.

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
`model_timeout`, `vllm_timeout` (engine and services), `default_rag` / `default_llm` (pair loaded when the
orchestrator starts), `llm_url`, `ollama_url`. The addresses of vLLM and of the services come from the sections
`vllm` and `vllm_reranker`.

The container has the socket of Docker and `ari_servers` at its path of the host: it starts and stops the RAG
containers, Ollama and vLLM with their `run.bash`, which mount the same folders as from the host. It does not use
the GPU and does not take the execution lock.

Known limits: the active pair is kept in memory only, so after a restart of the orchestrator `/status` says `idle`
until the next `POST /load` (which finds what is already running and takes a few seconds); the guard of
`sistema.sh` of high-level-unisa that stops engine and reranker under 15 GB of available memory is not ported.

| Path | Content |
|---|---|
| `orchestrator_server/server.py` | `Orchestrator` (the steps of a load, `SERVICES`), `OrchestratorService` (HTTP, on `HTTPService`) |
| `orchestrator_server/config.py` | `OrchestratorConfig` |
| `docker/` | image `ari_orchestrator`, `entrypoint.bash`, `run.bash` |

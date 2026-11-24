# LLM (+RAG) server

The answers of the demo over HTTP (`<LLM_PORT>`, default 8010): the custom actions of Rasa ask it instead of calling
ChatGPT themselves. The generation is done by a **backend**: the reference is ChatGPT, ported from the actions of
PepperDemo; a local model is another implementation of the same abstract class.

```
 client ──► rasa_proxy :5002 ──► social bot ──► action server ──HTTP POST /chat──► llm_server :8010 ──► backend
                                                 (llm_client.py)                   history, system prompt,     ChatGPT (OpenAI)
                                                                                   RAG, execution lock         transformers (TODO)
```

| Path | Content |
|---|---|
| `llm_server/backends/base.py` | `ChatBackend`: the **abstract class** (`load`, `generate`, `describe`), `Reply`, `BackendError`, the message format |
| `llm_server/backends/openai_chat.py` | `ChatGPTBackend` (`openai`): the class ChatGpt of the actions, the **reference implementation**. Text and images with Chat Completions, RAG with the Responses API and file_search, new attempt after server errors |
| `llm_server/backends/transformers_local.py` | `TransformersBackend` (`transformers`): local model on the GPU. TODO: port `chatmodel.py` of PepperDemo |
| `llm_server/backends/ollama.py` | `OllamaBackend` (`ollama`): a model of Ollama (`ollama/docker/run.bash`), native API with `num_ctx`; the generator of the RAG servers |
| `llm_server/backends/vllm.py` | `VllmBackend` (`vllm`): a model served by vLLM (`vllm/docker/run.bash`), OpenAI API `/v1/chat/completions`, `chat_template_kwargs.enable_thinking`; the generator of high_level_unisa (`qwen3.6-nvfp4`) |
| `llm_server/backends/echo.py` | `EchoBackend` (`echo`): answers with the request, for the tests without key or GPU |
| `llm_server/backends/__init__.py` | `BACKENDS`: the registry, `llm.backend` picks one |
| `llm_server/conversation.py` | history of every conversation (window of `history_length` messages) |
| `llm_server/rag.py` | `RagServers`: client of the RAG servers ([rag/](../rag/README.md)); `Retriever` of the server for the backends without native RAG. TODO: port `rag.py` of PepperDemo |
| `llm_server/server.py`, `config.py` | HTTP server, `LLMConfig` |
| `prompts/ari_system.txt` | default system prompt: the `PROMPT_SYSTEM` of ChatGpt |
| `docker/` | image `ari_llm_server` (base `ari_whisper_server:spark`, plus `openai`), `entrypoint.bash`, `run.bash` |

## A new backend
1. A subclass of `ChatBackend` in `llm_server/backends/<name>.py`: `name`, `native_rag` (it searches the documents
   itself, as file_search) and `vision` (it accepts images), `load()` (client or model, once at the start) and
   `generate(messages, *, rag, temperature, max_new_tokens, timeout) -> Reply`. `messages` are in the OpenAI chat
   format, system prompt first; on failure it raises `BackendError` (the server replies 502).
2. Register it in `BACKENDS` (`backends/__init__.py`).
3. `llm.backend: <name>` in `config/servers.yaml`. The server, its API and the actions of Rasa do not change.

The full checklist (options of a generator of a RAG, dependencies, orchestrator, tests) is in
[docs/integration.md → A new generation backend](../docs/integration.md#2-a-new-generation-backend-llm-server).

The backend is stateless: the server keeps the history, adds the system prompt and, for the backends without
`native_rag`, the chunks of its retriever.

## API
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service": "llm_server", "backend", "model", "native_rag", "vision", "rag", "loaded", "conversations"}` |
| `POST /image` | body: the image as bytes (`Content-Type` `image/webp`, `image/jpeg` or `image/png`), optional header `X-Image-Id` (32 hex characters, the id chosen by the client). Kept in memory 600 s. Reply `{"image_id", "bytes", "mime", "receive_s", "total_s"}`. ARI uploads here the camera frame of the questions about the scene, in parallel to the slots of Rasa; `image_slot` holds `image_id:<id>` and the action passes `image_id` to `/chat` |
| `POST /chat` with history | `{"conversation_id", "message"}`; optional `image_id` (of `POST /image`) or `image_base64` (JPEG or WebP), `system` (default: `llm.system_prompt_file`), `rag` (default: `llm.rag`; the name of a RAG server, e.g. `"department_office"`: see below), `temperature`, `max_new_tokens` |
| `POST /chat` one-shot | `{"messages": [{"role", "content"}...]}` or `{"prompt", "system"}`: no history |
| `"llm"` in `/chat` | `"<backend>:<model>"`, e.g. `"ollama:qwen3.6:35b-a3b"`: this model instead of `llm.backend`/`llm.model` (loaded by the [orchestrator](../orchestrator/README.md)); the backend is created at its first request and kept |
| reply | `{"text", "model", "served_model", "rag", "chunks", "conversation_id", "history", "queue_s", "rag_s", "generate_s", "total_s"}` |
| `POST /reset` | `{"conversation_id"}`: forgets its history |
| errors | `400` bad request, `501` backend not implemented, `502` the backend did not answer (OpenAI unreachable, key refused...), `500` other |

```bash
curl -X POST -H 'Content-Type: application/json' localhost:8010/chat \
  -d '{"conversation_id": "test", "message": "quali sono le soglie di emoglobina glicata?", "rag": true}'
```

The actions of Rasa use `conversation_id` = the sender of the tracker, so every user has their own history (in
PepperDemo all the users shared one). The history is in memory: a restart of the server forgets it.

## Configuration
| Where | What |
|---|---|
| `config/servers.yaml`, section `llm` | `backend` (`openai`, `transformers`, `echo`), `model` (`gpt-4o`), `temperature`, `timeout`, `system_prompt_file`, `max_char`, `history_length`, **`rag`** (default of the requests), the retriever of the local backends |
| `config/servers.yaml`, `llm.rag_servers`, `llm.rag_timeout` | the RAG servers (`<name>=<url>, ...`) and the seconds of their retrieval |
| `config/servers.yaml`, section `ollama` | backend `ollama`: `url`, `num_ctx`, `keep_alive`, `think` (a RAG server can give its own) |
| `config/servers.yaml`, section `vllm` | backend `vllm`: `url`, `enable_thinking` (a RAG server can give its own); the other keys are the settings of the vLLM engine, read by `vllm/docker/run.bash` |
| `config/servers.yaml`, section `openai` | parameters of ChatGPT: `vector_store_id` and the file_search settings of the RAG, `reasoning_effort`, `rag_model`, `max_retries`... |
| `config/secrets.env` (not versioned) | `OPENAI_API_KEY=sk-...` |
| environment | `LLM_<FIELD>` and `OPENAI_<FIELD>` override the YAML file, e.g. `LLM_RAG=true`, `LLM_BACKEND=echo` |

RAG with ChatGPT: `llm.rag: true` (or `"rag": true` in the request) makes the backend search the documents of the
OpenAI vector store before answering (Responses API, file_search); with `false` it answers with Chat Completions.
The social action of Rasa asks for the default of the server; the questions about the scene never use it, as in
PepperDemo. The behaviour trees of ARI (`LLMNode` of `ari_control`, with `LLM_RAG_RUNNING=True`) call `/chat` directly:
`"rag": true` for the open requests, the camera frame and `"rag": false` for the questions about the scene.

## RAG servers
`"rag": "<name>"` (a name of `llm.rag_servers`, e.g. `department_office=http://localhost:8030`) asks the RAG server
`<name>` for the context of the question (`POST /retrieve`, see [rag/README.md](../rag/README.md)), then answers with
the system prompt and the **model of that RAG** (its `generator`: e.g. backend `ollama`, `qwen3.6:35b-a3b-q8_0`, or
backend `vllm`, `qwen3.6-nvfp4` for high_level_unisa), not
with `llm.backend`. The backend of a RAG is created at its first request and kept. The history of the conversation
holds the questions, not the documents. The reply has also `rag_info` (filters, calls of the registers, sources);
`rag_s` is the time of the RAG server, its queue included. `"rag": true/false` work as before.

```bash
curl -X POST -H 'Content-Type: application/json' localhost:8010/chat \
  -d '{"conversation_id": "test", "message": "chi era presente nella seduta del 29/09/2025?", "rag": "department_office"}'
```
The RAG server takes the execution lock during the retrieval, this server during the generation, after it.

## Run with Docker (Spark)
From `ari_servers`:
```bash
echo 'OPENAI_API_KEY=sk-...' > config/secrets.env && chmod 600 config/secrets.env   # once
bash llm/docker/run.bash build       # image ari_llm_server, once (needs ari_whisper_server:spark)
bash llm/docker/run.bash start       # container llm_server, restarts with the machine
bash llm/docker/run.bash status      # /health
bash llm/docker/run.bash test "qual è la ricetta della pizza?"
bash llm/docker/run.bash logs
bash llm/docker/run.bash stop
LLM_BACKEND=echo bash llm/docker/run.bash start   # without key, for the tests
```
After a change of the configuration or of the code: `stop`, then `start`. Start it before the RASA server: without
it the social bot answers with its courtesy phrase.

Without Docker: `export PYTHONPATH=$PWD/common:$PWD/llm` and `python -m llm_server` (needs `openai` for ChatGPT).

## Execution lock
The server holds the lock of the servers (no concurrent executions) during the generation, also with ChatGPT,
where the model runs at OpenAI: Whisper waits for it (`queue_s`). Since the actions of Rasa call this server,
`rasa.exclusive` is `false`: with the lock held by the RASA proxy, this server would wait for it forever.

Every request writes a `llm_server.done` line (and `openai.request` / `openai.reply` with ChatGPT) in
`latency_logs/latency_<date>_<host>.csv`.

## Tested on 2026-11-12
- `echo` backend, host and container: history per conversation (window of 4 messages), reset, one-shot requests,
  images, errors 400; through Rasa (`:5002` → action → LLM server) with separate histories per sender.
- `openai` backend: both paths reach OpenAI (Chat Completions without RAG, Responses API with RAG); the key of
  `config/secrets.env` was refused (401), so the server replied 502 and Rasa its courtesy phrase. **The answers of
  ChatGPT are still to be checked with a valid key.**

## Next steps
1. `TransformersBackend`: load and generate as `ChatModel` of PepperDemo (model in `models/llm/<name>`).
2. `FaissRetriever`: index of `prepare_rag.py`, BM25 + FAISS, reranker, as `Rag` of PepperDemo.

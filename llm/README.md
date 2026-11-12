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
| `llm_server/backends/echo.py` | `EchoBackend` (`echo`): answers with the request, for the tests without key or GPU |
| `llm_server/backends/__init__.py` | `BACKENDS`: the registry, `llm.backend` picks one |
| `llm_server/conversation.py` | history of every conversation (window of `history_length` messages) |
| `llm_server/rag.py` | `Retriever` of the server for the backends without native RAG. TODO: port `rag.py` of PepperDemo |
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

The backend is stateless: the server keeps the history, adds the system prompt and, for the backends without
`native_rag`, the chunks of its retriever.

## API
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service": "llm_server", "backend", "model", "native_rag", "vision", "rag", "loaded", "conversations"}` |
| `POST /chat` with history | `{"conversation_id", "message"}`; optional `image_base64` (JPEG), `system` (default: `llm.system_prompt_file`), `rag` (default: `llm.rag`), `temperature`, `max_new_tokens` |
| `POST /chat` one-shot | `{"messages": [{"role", "content"}...]}` or `{"prompt", "system"}`: no history |
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
| `config/servers.yaml`, section `openai` | parameters of ChatGPT: `vector_store_id` and the file_search settings of the RAG, `reasoning_effort`, `rag_model`, `max_retries`... |
| `config/secrets.env` (not versioned) | `OPENAI_API_KEY=sk-...` |
| environment | `LLM_<FIELD>` and `OPENAI_<FIELD>` override the YAML file, e.g. `LLM_RAG=true`, `LLM_BACKEND=echo` |

RAG with ChatGPT: `llm.rag: true` (or `"rag": true` in the request) makes the backend search the documents of the
OpenAI vector store before answering (Responses API, file_search); with `false` it answers with Chat Completions.
The social action of Rasa asks for the default of the server; the questions about the scene and the knowledge
graph never use it, as in PepperDemo.

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

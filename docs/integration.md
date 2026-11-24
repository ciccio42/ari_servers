# Integration guide

How the servers of `ari_servers` talk to each other, and the steps to add a **generation backend** to the LLM server
or a **RAG** (knowledge base + retrieval). The details of each server are in its README: see
[References](#references).

## Components

| Component | Port | Code | Role |
|---|---|---|---|
| Whisper | 8000 | [whisper/](../whisper/README.md) | speech to text |
| LLM server | 8010 | [llm/](../llm/README.md) | the answers: history of the conversations, system prompt, generation with a **backend** (`openai`, `ollama`, `vllm`, `echo`), the RAG servers |
| TTS | 8020 | [tts/](../tts/README.md) | text to speech (Piper) |
| RAG servers | 8030-8032 | [rag/](../rag/README.md) | the retrieval of one knowledge base each, in its own container and conda environment |
| Ollama | 11500 (localhost) | [ollama/docker/run.bash](../ollama/docker/run.bash) | the local models of department_office and department |
| vLLM engine | 8001 (localhost) | [vllm/docker/run.bash](../vllm/docker/run.bash) `engine` | the model of high_level_unisa (`qwen3.6-nvfp4`, vLLM 0.31 with MTP) |
| vLLM reranker | 8002 (localhost) | [vllm/docker/run.bash](../vllm/docker/run.bash) `reranker` | a **service** of high_level_unisa (`bge-reranker-v2-m3`, `/rerank`) |
| Orchestrator | 8040 | [orchestrator/](../orchestrator/README.md) | loads one pair LLM + RAG at a time in memory: RAG container, model (Ollama or vLLM engine), services of the RAG |
| RASA | 5002-5004 | [rasa/](../rasa/README.md) | dialogue; its custom actions call the LLM server |
| shared code | | [common/ari_servers/](../common/ari_servers) | `config.py`, `http.py` (`HTTPService`), `execution.py` (lock), `latency_log.py`, `rag.py` (interface of the RAG servers) |

All of them are configured by [config/servers.yaml](../config/servers.yaml) (one section per server, overridden by
`<PREFIX>_<FIELD>` and `--<field>`) and mount `ari_servers` as `/workspace`: see
[README.md → Configuration](../README.md#configuration).

## 1. Communication flow

### The whole demo
```
 ARI ──audio──► whisper :8000 ──text──► ARI
 ARI ──text───► rasa :5002-5004 ──► bot ──► custom action ──POST /chat──┐
 ARI ──POST /chat (behaviour trees, LLMNode)─────────────────────────────┤
                                                                        ▼
                         orchestrator :8040 ──(load/unload)──► llm_server :8010
                         one pair LLM + RAG                      │  "rag": false/true ──► ChatGPT (+ OpenAI vector store)
                         in memory                               │  "llm": "<backend>:<model>" ──► that model
                                                                 │  "rag": "<name>" ─POST /retrieve─► rag_<name> :803x
                                                                 │                   ◄─ context, prompt, generator
                                                                 │                      (services of the RAG, e.g.
                                                                 │                       vllm_reranker :8002)
                                                                 └─► generation ──► ollama :11500 or vllm :8001
 ARI ◄──text── llm_server ;  ARI ──text──► tts :8020 ──WAV──► ARI
```
Everything is HTTP + JSON on the same machine (Docker network host): the calls between servers cost a few
milliseconds, the time goes into retrieval and generation (measured: [README.md → Execution flow](../README.md#execution-flow)).

### Choosing the pair (orchestrator)
Before the questions, the client loads the pair of the demo:
```
POST :8040/load {"rag": "department_office"}       (or {"rag": "high_level_unisa"}, {"llm": "ollama:qwen3.6:35b-a3b"},
                                                    {"llm": "openai:gpt-4o"})
  1. stops the containers of the other RAGs (and vLLM, the services, when the new pair does not use them)
  2. starts rag_department_office (rag/docker/run.bash) and waits for GET /health "loaded": true
  3. reads from its /health the generator: "ollama:qwen3.6:35b-a3b-q8_0", options {"num_ctx": 16384}, and the
     services it needs: [] (high_level_unisa: generator "vllm:qwen3.6-nvfp4", services ["vllm_reranker"])
  4. starts Ollama if needed, unloads the other models, loads this one with that num_ctx (keep_alive 2 h);
     for a generator on vLLM: starts the vLLM engine and waits until it serves the model
  5. starts the services of the RAG (vllm/docker/run.bash start reranker) and waits until they answer
```
One pair at a time: the memory of the Spark does not hold every model. `GET :8040/status` tells what is loaded.
Details: [orchestrator/README.md](../orchestrator/README.md).

### A question with a RAG
```
client ──POST :8010/chat {"conversation_id", "message", "rag": "<name>"}──► llm_server
  1. llm_server: the message goes to the history of conversation_id (window llm.history_length)
  2. llm_server ──POST :803x/retrieve {"question", "history": [previous turns]}──► rag_<name>
       the RAG server takes the execution lock, runs its retrieval (filters, search, reranking, its own model
       calls on Ollama if any) and replies:
       {"chunks", "system", "user", "answer", "drop_patterns", "retrieval_id", "generator", "queue_s", "retrieve_s", ...}
  3. "answer" not empty (greeting, out of domain, nothing found): that is the reply, no generation
  4. llm_server takes the execution lock and generates with the generator of the RAG
       (backend created at its first request and kept): [system] + history + [user]
  5. drop_patterns removed from the text
  6. "retrieval_id" not empty: llm_server ──POST :803x/postprocess {"retrieval_id", "answer"}──► rag_<name>
       (e.g. high_level_unisa checks the names against the context) ◄── {"text"}
  7. the history keeps question and answer (not the documents)
client ◄── {"text", "model", "rag", "chunks", "rag_info": {extra fields of the RAG}, "history",
            "queue_s", "rag_s", "generate_s", "total_s"}
```
Without `"rag": "<name>"`: steps 2-3 and 5-6 do not exist; the generation uses `llm.backend` or the `"llm"` of the
request. API of the LLM server: [llm/README.md → API](../llm/README.md#api); of the RAG servers:
[rag/README.md → API](../rag/README.md#api-of-a-rag-server).

### Rules
- **Execution lock** (`run/execution.lock`): one request at a time among all the servers. Never hold it while
  calling another server that takes it: the RAG server holds it during the retrieval and releases it, then the LLM
  server takes it for the generation; the RASA proxy does not take it. Ollama and vLLM do not take it (their callers
  do).
  [README.md → No concurrent executions](../README.md#no-concurrent-executions).
- **Same model, same `num_ctx`**: a different `num_ctx` makes Ollama load the model again (tens of seconds). The
  model calls inside a RAG, the generator options it declares and the orchestrator use the same value. vLLM has no
  `num_ctx`: its limit is `max_model_len` of the engine.
- **The model of a RAG is the one it was evaluated with**: the RAG declares it (`generator_*` in its section); the
  LLM server and the orchestrator follow it.
- **Latency log**: every server writes `mark(...)` lines in `latency_logs/latency_<date>_<host>.csv`
  (`llm_server.done`, `rag_<name>.retrieve`, `rag_<name>.postprocess`, `ollama.request/reply`, `vllm.request/reply`,
  `orchestrator.load/ready`).
- **Errors**: 400 bad request, 501 models not loaded, 502 the next server did not answer (with the reason, e.g. a RAG
  not loaded: load it with the orchestrator). A failed request never stops a server.

## 2. A new generation backend (LLM server)

A backend turns messages into an answer: ChatGPT (`openai`), Ollama (`ollama`), vLLM (`vllm`, OpenAI-compatible API),
the tests (`echo`); `transformers` is still to port. Interface: [llm_server/backends/base.py](../llm/llm_server/backends/base.py).

1. **Class**: `llm/llm_server/backends/<file>.py`, a subclass of `ChatBackend`:
   - `name = "<backend>"` (the key in `llm.backend`, in `"llm": "<backend>:<model>"` and in the generator of a RAG),
     `native_rag` (it searches documents itself, as OpenAI file_search), `vision` (it accepts images);
   - `__init__(self, cfg, options=None)` if it can be the generator of a RAG: `options` are the options the RAG
     declares (e.g. `num_ctx`), they override the section of the backend in servers.yaml;
   - `load()`: client or model, once (the LLM server calls it at the start for `llm.backend`, at the first request
     for the others);
   - `generate(messages, *, rag, temperature, max_new_tokens, timeout) -> Reply`: `messages` in the OpenAI chat
     format, system first; stateless; on failure raise `BackendError` (the server replies 502);
   - `describe()`: extra fields of `/health`;
   - its parameters: a dataclass loaded with `config.load(<Config>, "<section>", "<PREFIX>", argv=[])`, as
     `OllamaConfig` in [ollama.py](../llm/llm_server/backends/ollama.py) or `OpenAIConfig` in
     [openai_chat.py](../llm/llm_server/backends/openai_chat.py); `mark("<backend>", "request"/"reply", ...)` for
     the latency log.
2. **Register** it in `BACKENDS` ([backends/\_\_init\_\_.py](../llm/llm_server/backends/__init__.py)). The
   orchestrator reads the backend names from the `name = "..."` of these files: nothing to change there.
3. **Dependencies**: the image `ari_llm_server` ([llm/docker/Dockerfile](../llm/docker/Dockerfile)); then
   `bash llm/docker/run.bash build`.
4. **Configuration**: a section for its parameters in [config/servers.yaml](../config/servers.yaml) (keys in
   `config/secrets.env`, never in a versioned file); `llm.backend: <backend>` to make it the default.
5. **Run and test**: `bash run_servers.bash restart llm`; one request with `"llm": "<backend>:<model>"`
   (`bash llm/docker/run.bash test`, or `curl :8010/chat`); if it loads a local model, through the orchestrator
   (`load llm=<backend>:<model>`; today the orchestrator loads the models of `ollama` and starts the engine of
   `vllm`: a backend with another server needs its start and stop in `Orchestrator.load`, as `start_vllm`).
6. **Docs**: the table of [llm/README.md](../llm/README.md) and, if it is the generator of a RAG, its README.

## 3. A new RAG

A RAG server does only the retrieval: the answer is generated by the LLM server with the prompt and the model of the
RAG. The three RAGs already integrated are the examples, from the simplest:
[department](../rag/department/README.md) (dense search, model calls for filter, rewriting and router),
[department_office](../rag/department_office/README.md) (Qdrant, registers with tool calls),
[high_level_unisa](../rag/high_level_unisa/README.md) (history, postprocess, generator on vLLM, a service). Interface:
[common/ari_servers/rag.py](../common/ari_servers/rag.py).

### 3.1 Analysis of the original repository
- Find the **code that answers** (often one class with a `retrieve_and_generate`/`rispondi` method) and separate it
  from what builds the knowledge base (scraping, conversion, chunking, indexing), from the tests and from the
  interfaces (bot Telegram, Gradio): only the first is ported.
- Find the **final configuration** (switches, models, `.env`, notes of the authors) and the **model** of the
  generation and of the internal calls (router, rewriting, filters, planners).
- Find the **knowledge base in use** (index, registries, JSON files) and compare it with the copy in
  `rag_knowledge_base/<name>` (`cmp`, number of chunks per collection).
- Note where the pipeline uses the **history** and what it does **after** the generation.

### 3.2 Steps
1. **Knowledge base**: copy it in `rag_knowledge_base/<name>/` (not versioned), with every data file the code reads.
2. **Model**: the one of the original, on the engine of its final configuration: Ollama (`bash ollama/docker/run.bash
   pull <model>`; a Hugging Face model: the same model on the Ollama library, fp16 for the same weights) or vLLM
   (section `vllm`, `bash vllm/docker/run.bash download engine`). Read the documents of the authors (reports, launch
   scripts): the final configuration may be newer than the code folder named "final". Check the memory: the
   orchestrator keeps one pair at a time.
3. **Code**, `rag/<name>/<name>_rag/`:
   - `utils/`: the modules of the original that answer, **copied**, with a header that says where they come from
     and what changed. Change only what is needed: relative imports, no `.env` (paths and switches from the
     environment, set by `rag.py`), the internal model calls to the engine of the generator (Ollama: `base_url`, the
     `num_ctx` of the generator; vLLM: the OpenAI base with `/v1`), the pieces of the chain kept on the instance (e.g.
     at the end of `_build_pipeline`);
   - `config.py`: a dataclass extending `RagConfig` (`port`, `kb_dir`, `generator_backend`, `generator_model`,
     `temperature`, `max_new_tokens`, the options of the model, the switches of the original), loaded with
     `config.load(<Config>, "rag_<name>", "RAG_<NAME>")`;
   - `rag.py`: a subclass of `Rag`:
     - `load()`: imports `utils` and builds the engine (index, embedding model, reranker);
     - `retrieve(question, history)`: the steps of the original **up to the context**, without the generation.
       It returns a `Retrieval`: `chunks` (`Chunk(text, source, score)`), `system` (the system prompt of the
       answer), `user` (context and question, the human message of the original prompt), and, when they apply:
       `answer` (a reply without the model: greeting, out of domain, nothing found), `drop_patterns` (regular
       expressions removed from the answer), `info` (extra fields: path, filters, rewritten question...),
       `postprocess=True` with `state` (what `postprocess` needs, e.g. the context);
     - `postprocess(answer, state)` only if the original changes the answer after the generation;
     - `generator()`: `Generator(backend, model, temperature, max_new_tokens, {"num_ctx": ...})`;
     - `describe()`: the fields of `/health`;
   - `server.py` and `__main__.py`: as [department/server.py](../rag/department/department_rag/server.py)
     (`RagService(rag, execution.from_config())`), `__init__.py` and `utils/__init__.py` with a docstring.
4. **Dependencies**: `rag/<name>/environment.yml`, `name: rag_<name>`, with the versions of the original
   (its `requirements.txt` / `pip freeze`), torch from the index `cu130` (GB10), `pyyaml`; only what the retrieval
   uses. `bash rag/docker/run.bash build`: the image `ari_rag_server` gets one more conda environment.
5. **Configuration**, [config/servers.yaml](../config/servers.yaml): the section `rag_<name>` (a free port, `kb_dir`,
   generator, options), `<name>` in `rag_servers`, `<name>=http://localhost:<port>` in `llm.rag_servers`;
   `bash run_servers.bash restart llm` (the LLM server reads `rag_servers` at the start). The orchestrator finds the
   RAG by itself (a folder `rag/<name>` with `environment.yml`).
6. **Run**: `bash orchestrator/docker/run.bash load rag=<name>`, then `status`. Logs:
   `bash rag/docker/run.bash logs <name>`. If the start is slow, measure its phases before optimizing.
   **Services**: when the RAG needs another server (e.g. a reranker served by vLLM), declare it in `"services"` of its
   `/health` (`describe()`) and add it to `Orchestrator.SERVICES` (its `run.bash`, container, section with the `url`):
   the orchestrator starts it with the pair and stops it with the next one.
7. **Test**:
   - only the retrieval: `bash rag/docker/run.bash test <name> "<question>"`;
   - the whole chain: `curl :8010/chat` with `"rag": "<name>"`, on questions of the test banks of the original,
     next to the reference answers and, if the original saved them, its answers (same text expected with
     temperature 0, apart from the date in the prompt);
   - greeting, out of domain, a follow-up question (history), a question with nothing in the knowledge base;
   - the other RAGs still answer (`load` them and ask one question each).
8. **Docs**: `rag/<name>/README.md` (what it answers, knowledge base, models, flow of a question, table "Here ↔
   original", differences from the original, configuration); a row in the tables of [rag/README.md](../rag/README.md),
   [README.md](../README.md) and [orchestrator/README.md](../orchestrator/README.md).

### 3.3 Choices already made (keep them consistent)
| Question | Choice |
|---|---|
| Who generates the answer | the LLM server, with the model of the RAG: never a second copy of the model in the RAG |
| Model calls inside the retrieval (router, rewriting, planner) | directly to the engine of the generator (Ollama or vLLM) from the RAG server, same model (and `num_ctx`) as the generator |
| Other servers of a RAG (reranker...) | services: one container each, declared by the RAG in `/health`, started and stopped by the orchestrator |
| History of the conversation | kept by the LLM server; the RAG receives it in `/retrieve` and uses it only where the original does |
| Logic of the original | unchanged, also its known limits (documented in the README of the RAG) |
| Knowledge base | built by the original repository, copied in `rag_knowledge_base/<name>`; nothing of the build is ported |
| Container and environment | one container `rag_<name>` and one conda environment `rag_<name>` per RAG, one image for all |

## References
| Document | Content |
|---|---|
| [README.md](../README.md) | overview, execution flow, run all the servers, configuration, execution lock, shared folders, knowledge bases |
| [llm/README.md](../llm/README.md) | LLM server: backends, API, RAG servers, `"llm"` in the requests, configuration, Docker |
| [rag/README.md](../rag/README.md) | RAG servers: API (`/retrieve`, `/postprocess`), run, list of the RAGs |
| [rag/department_office/README.md](../rag/department_office/README.md) | RAG of the department resolutions (rag_unisa) |
| [rag/high_level_unisa/README.md](../rag/high_level_unisa/README.md) | RAG of the Ateneo (high-level-unisa, final configuration): history, postprocess, vLLM engine and reranker |
| [vllm/docker/run.bash](../vllm/docker/run.bash) | vLLM engine and reranker: settings and memory checks (header) |
| [rag/department/README.md](../rag/department/README.md) | RAG of the DIEM department (DIEM_RAG) |
| [orchestrator/README.md](../orchestrator/README.md) | orchestrator: pairs LLM + RAG, API, measured times and memory |
| [whisper/README.md](../whisper/README.md) | Whisper server |
| [tts/README.md](../tts/README.md) | TTS server (Piper) |
| [rasa/README.md](../rasa/README.md) | RASA server and bots |
| [audio_files/REPORT_modelli_whisper.md](../audio_files/REPORT_modelli_whisper.md) | report of the models (Whisper, LLM, Rasa times) |
| [common/ari_servers/rag.py](../common/ari_servers/rag.py) | interface of the RAG servers (docstring: API and contract) |
| [llm/llm_server/backends/base.py](../llm/llm_server/backends/base.py) | interface of the generation backends |

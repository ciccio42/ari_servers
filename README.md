# ARI-Demo Servers

Network servers of the demos of ARIS: they run on the machine with the GPU (DGX Spark, Jetson, cluster)
and the robot reaches them over HTTP. The clients of these servers stay in `ari_control`.

Architecture: [ari_demo_arch.png](ari_demo_arch.png). This repository is the **DGX-Spark** box: the HTTP servers, the
RAGs with their local models and the rule *no concurrent executions*. Visual-Perception (Jetson), Behavioral-Trees,
HTTP clients and Robot Control (Main-PC) are in `ari_control`.

| Server | Folder | Port (`<..._PORT>`) | State |
|---|---|---|---|
| Whisper (speech to text) | [whisper/](whisper/README.md) | 8000 | model as a parameter, tested on the Spark |
| LLM (+RAG) (answers) | [llm/](llm/README.md) | 8010 | backends: ChatGPT (default of the demo, with the OpenAI RAG), Ollama (the RAGs of `rag/`), echo (tests) |
| TTS (Piper, audio of the answers) | [tts/](tts/README.md) | 8020 | Piper on the GPU, client: `TtsNode` of the behaviour trees |
| RAG (retrieval of a knowledge base) | [rag/](rag/README.md) | 8030 (`department_office`), 8031 (`high_level_unisa`), 8032 (`department`), one per RAG | one container and one conda environment per RAG, asked by the LLM server |
| Ollama (local models of the RAGs) | [ollama/docker/](ollama/docker/run.bash) | 11500 (localhost only) | `qwen3.6:35b-a3b-q8_0` (`department_office`), `qwen3.6:35b-a3b` (`high_level_unisa`), `qwen3:30b-a3b-instruct-2507-fp16` (`department`) |
| RASA (dialogue) | [rasa/](rasa/README.md) | 5002, 5003, 5004 (one per bot) | the 3 bots of PepperDemo and their actions behind the proxy with the lock |
| Orchestrator (models in memory) | [orchestrator/](orchestrator/README.md) | 8040 | loads on request one pair LLM + RAG at a time (RAG container, model of Ollama), unloading the previous one |

```
ari_servers/
├── config/servers.yaml   ports, models and options of every server, RAGs, execution lock
├── common/ari_servers/   shared code: config.py, http.py (HTTPService), execution.py (lock), latency_log.py,
│                         rag.py (interface of the RAG servers)
├── whisper/              whisper_server/ (package), docker/, conda/, tools/
├── llm/                  llm_server/ (package): backends/ (abstract class, ChatGPT, Ollama, echo), server.py, rag.py; docker/
├── tts/                  tts_server/ (package, Piper), docker/
├── rag/                  one folder per RAG: <name>/environment.yml, <name>/<name>_rag/ (server.py, rag.py, config.py,
│                         utils/); docker/ (image ari_rag_server, one conda environment per RAG)
├── ollama/docker/        Ollama in Docker, models in models/ollama
├── orchestrator/         orchestrator_server/ (package): one pair LLM + RAG in memory at a time; docker/
├── rasa/                 bots/ (social, rock_paper_scissor, connect_four), rasa_proxy/ (package), docker/
├── rag_knowledge_base/   knowledge bases of the RAGs (not versioned)
├── models/  audio_files/  latency_logs/  run/
└── run_servers.bash      starts, stops and checks all the servers
```

## Execution flow
Before the questions, the client chooses the pair LLM + RAG of the demo with the orchestrator: `POST :8040/load
{"rag": "high_level_unisa"}` (or `{"llm": "ollama:qwen3.6:35b-a3b"}`, `{"llm": "openai:gpt-4o"}`). It starts that RAG
and loads its model, stopping the previous pair: one pair at a time in memory ([orchestrator/](orchestrator/README.md)).

ARI (behaviour trees of `ari_control`) sends the audio of the user to Whisper, the text to Rasa or directly to the
LLM server, and the answer to the TTS server. The answers of the LLM server come from ChatGPT, or from a local RAG
when the request names it:

```
 ARI ──audio──► whisper_server :8000 ──text──► ARI
 ARI ──text───► rasa_proxy :5002-5004 ──► bot ──► custom action ──POST /chat──┐
 ARI ──POST /chat (LLMNode, RAG requests)─────────────────────────────────────┤
                                                                              ▼
                                                                      llm_server :8010
           "rag": false / true (ChatGPT + OpenAI vector store) ◄──────────┤
           "rag": "<name>" ───────POST /retrieve──► rag_<name> :8030 ─────┤  context, system prompt, generator
                                                     Qdrant, BGE-M3,      │
                                                     reranker, registers  │
                                                     (planner ──► Ollama) │
           generation with the generator of the RAG ──► ollama_server :11500 (qwen3.6:35b-a3b-q8_0)
 ARI ◄──text── llm_server ;  ARI ──text──► tts_server :8020 ──WAV──► ARI
```

A request to the LLM server with a RAG, step by step:

1. `POST /chat {"conversation_id", "message", "rag": "department_office"}`: the LLM server adds the message to the
   history of the conversation.
2. It asks the RAG server for the context: `POST /retrieve {"question", "history"}` (high_level_unisa rewrites the
   question with the history; department_office uses only the question). The RAG server takes the execution lock,
   deduces the filters from the question (year, date, type of document), asks the registers when the question is
   about many meetings (the planner calls Ollama with tool calls), searches Qdrant (dense + sparse, reranker), and
   returns the chunks, its system prompt, the request for the model (documents and question) and its generator
   (`ollama`, `qwen3.6:35b-a3b-q8_0`, temperature, tokens, `num_ctx`). With nothing retrieved it returns the answer
   itself, without the model.
3. The LLM server takes the execution lock and generates with the generator of the RAG (backend `ollama`, created at
   the first request of that RAG), with the history of the conversation; it removes the citations of the registers.
   When the RAG returned a `retrieval_id`, the LLM server sends it the answer (`POST /postprocess`): high_level_unisa
   corrects the names copied wrong by the model, checking them against the context.
4. Reply: `{"text", "model", "rag", "chunks", "rag_info": {"filters", "registers", "sources"}, "queue_s", "rag_s",
   "generate_s", "total_s"}`. The history keeps the question and the answer, not the documents.

Without `"rag": "<name>"` nothing changes: ChatGPT with `llm.backend` (the social bot of Rasa, the questions about the
scene). Measured on the Spark (2026-11-24): 8-21 s per answer of `department_office` (retrieval 3-10 s, generation
3-11 s); the first request after 2 hours without requests loads qwen3.6 (38 GB, about 55 s).

## Run all the servers
[run_servers.bash](run_servers.bash) runs the servers in Docker on the Spark, in order (Whisper, Ollama, LLM, TTS, RASA,
orchestrator: the RASA server asks the LLM server, the LLM server the RAGs and Ollama), and waits until each one
answers. The RAGs are started by the orchestrator, one pair LLM + RAG at a time (`bash run_servers.bash start rag`
starts all of them, if the memory allows). From `ari_servers`:
```bash
echo 'OPENAI_API_KEY=sk-...' > config/secrets.env && chmod 600 config/secrets.env   # once: key of the LLM server
bash run_servers.bash build          # once: the images (Whisper, LLM, TTS, RASA, RAG with its conda environments, orchestrator)
bash run_servers.bash start ollama && bash ollama/docker/run.bash pull \
    && bash ollama/docker/run.bash pull qwen3.6:35b-a3b \
    && bash ollama/docker/run.bash pull qwen3:30b-a3b-instruct-2507-fp16   # once: the models of the RAGs (38 + 23 + 61 GB)
bash run_servers.bash start          # all the servers but the RAGs; already running ones are left as they are
bash orchestrator/docker/run.bash load rag=high_level_unisa   # the pair of the demo: RAG + its model
bash run_servers.bash status         # container state and /health of every server
bash run_servers.bash restart        # after a change of the configuration
bash run_servers.bash stop
bash run_servers.bash restart llm    # one or more servers: whisper, ollama, llm, tts, rag (all the RAGs) or rag_<name>, rasa, orchestrator
bash run_servers.bash logs rag_department_office   # follows the log of one server
```

| Configuration file | What | Versioned |
|---|---|---|
| [config/servers.yaml](config/servers.yaml) | ports, models, RAG on/off, OpenAI and Ollama parameters, RAG servers, execution lock of all the servers | yes |
| [config/rasa.env](config/rasa.env) | language, address of the LLM server, flags of the demo | yes |
| `config/secrets.env` | `OPENAI_API_KEY=...` | **no** |

`-c <file>` uses another configuration instead of `config/servers.yaml`, e.g. one copy per demo or per machine:
`bash run_servers.bash -c config/servers_rag.yaml restart`. The file must be inside `ari_servers` (it is mounted in
the containers). `RASA_ENV_FILE=<file>` does the same for `rasa.env`. A variable exported in the shell overrides the
files for that run, e.g. `LLM_RAG=true bash run_servers.bash restart llm`. The containers restart with the machine;
a change of the configuration needs `restart`.

Ready after 15 s (Whisper, model already exported; the first start exports it, minutes), 3 s (LLM), 12 s (RASA),
about 1 minute (a RAG; the first start downloads its embedding model and reranker, about 10 minutes).
Each server has also its own `run.bash` (`whisper/docker`, `llm/docker`, `tts/docker`, `rasa/docker`, `rag/docker`,
`ollama/docker`), with `status`, `logs` and, for most of them, `test` and `shell`.

## RAG servers
A RAG is a knowledge base with its retrieval, in its own folder `rag/<name>` with the same structure: the server
interface (`server.py`, on `RagService` of [common/ari_servers/rag.py](common/ari_servers/rag.py)), the class of the
RAG (`rag.py`, a subclass of `Rag`), its configuration (`config.py`, section `rag_<name>` of servers.yaml) and its
code (`utils/`). Every RAG has its own conda environment (`rag/<name>/environment.yml`, built into the image
`ari_rag_server`) and its own container `rag_<name>`, so RAGs with different dependencies run at the same time.

| RAG | Port | Knowledge base | Generator |
|---|---|---|---|
| [department_office](rag/department_office/README.md) | 8030 | resolutions of the Consiglio Didattico di Ingegneria Informatica (rag_unisa, M.S. D'Acunto) | Ollama, `qwen3.6:35b-a3b-q8_0` |
| [high_level_unisa](rag/high_level_unisa/README.md) | 8031 | University of Salerno, whole Ateneo: bodies, people, offices, regulations (high-level-unisa, M.S. Grimaldi, Iannone) | Ollama, `qwen3.6:35b-a3b` |
| [department](rag/department/README.md) | 8032 | DIEM department: courses, people, research, calls, Erasmus (DIEM_RAG, course Generative AI) | Ollama, `qwen3:30b-a3b-instruct-2507-fp16` |

```bash
bash rag/docker/run.bash test department_office "chi era presente nella seduta del 29/09/2025?"   # only retrieval
curl -X POST -H 'Content-Type: application/json' localhost:8010/chat \
  -d '{"conversation_id": "test", "message": "quante volte è stato assente il prof. Auletta nel 2025?", "rag": "department_office"}'
```
In servers.yaml: `rag_servers` (the RAGs started by run_servers.bash), `rag_<name>` (port, `kb_dir`, generator,
retrieval options), `llm.rag_servers` (`<name>=<url>`: the RAGs the LLM server can ask), `ollama` (address and
options of Ollama). A new RAG: [rag/README.md](rag/README.md#a-new-rag).

## Configuration
Every server is a dataclass of options (e.g. `WhisperConfig`), filled from the lowest to the highest priority by:
its defaults, its section of [config/servers.yaml](config/servers.yaml), the environment `<PREFIX>_<FIELD>`
(`WHISPER_MODEL`, `LLM_PORT`, `RASA_ROUTES`, `RAG_DEPARTMENT_OFFICE_TOP_K`...), the command line `--<field>`. The
**model** of each server is a field: change it in the YAML file, or for one run with a variable or an option.
```bash
WHISPER_MODEL=openai/whisper-large-v3 python -m whisper_server       # another Whisper model
python -m whisper_server --model openai/whisper-small --optimization none --port 8001
```
`ARI_SERVERS_CONFIG` points to another YAML file.

The RASA server has a second file, [config/rasa.env](config/rasa.env) (language, address of the LLM server, flags of the
demo). Model, RAG on/off and OpenAI parameters of the answers are in the sections `llm` and `openai` of servers.yaml;
the OpenAI key only in `config/secrets.env`, which is not versioned. See [llm/README.md](llm/README.md#configuration).

## No concurrent executions
The servers share a lock file (`execution_lock` in servers.yaml, default `run/execution.lock`, `flock`): one request
at a time among all of them, also in different containers of the same machine (they all mount this folder as
`/workspace`). The time waited is `queue_s` in the replies. Empty `execution_lock`: disabled.

Do not chain two servers holding the lock: the custom actions of Rasa call the LLM server, so the RASA proxy does not
take it (`rasa.exclusive: false`) and the LLM server takes it during the generation. With a RAG, the RAG server takes
it during the retrieval and releases it, then the LLM server takes it during the generation: the LLM server never
holds it while it waits for the RAG server. Ollama does not take it (the server that calls it does).

## Run without containers
```bash
export PYTHONPATH=$PWD/common:$PWD/whisper:$PWD/llm:$PWD/rasa
python -m whisper_server        # needs the libraries of whisper (images or conda environment)
python -m llm_server --skip-load
python -m rasa_proxy            # standard library only
PYTHONPATH=$PWD/common:$PWD/rag/department_office python -m department_office_rag   # in the conda environment rag_department_office
```

Shared folders (the workspace of every server, mounted as `/workspace` in the containers):

| Folder | Content |
|---|---|
| `config/` | `servers.yaml`, `rasa.env`; `secrets.env` with the keys, not versioned |
| `models/` | models: `silero/silero_vad.onnx` (versioned), `whisper/` (exported at the first start), `rasa/` (trained bots), `ollama/` (models of Ollama), `rag/.hf_cache` (Hugging Face models of the RAGs): not versioned |
| `rag_knowledge_base/` | knowledge bases of the RAGs, not versioned (see below) |
| `audio_files/` | the 5 Italian WAV of the tests, their reference transcripts (`references.csv`) and the [report of the models](audio_files/REPORT_modelli_whisper.md) (Whisper, LLM, Rasa times) |
| `latency_logs/` | latency log of the servers (`latency_<date>_<host>.csv`), not versioned |
| `run/` | execution lock, not versioned |

## RAG knowledge bases
`rag_knowledge_base/` (not versioned) contains the knowledge bases; the servers that use them are in [rag/](rag/README.md):

| Folder | Content | Server |
|---|---|---|
| `high_level_unisa` | general questions on UNISA (built by M.S. Grimaldi, optimized code by Iannone): `diem_vectorstore/` (Chroma), `anagrafe.json`, `registro_strutture.json`, `posizioni_parent.json` | [rag/high_level_unisa](rag/high_level_unisa/README.md) |
| `department_office` | department resolutions (built by M.S. D'Acunto): `qdrant/`, `registers.json`, `chunks/` | [rag/department_office](rag/department_office/README.md) |
| `department` | DIEM department (course Generative AI, local version): `vectorstore/` (Chroma, 13 collections) | [rag/department](rag/department/README.md) |

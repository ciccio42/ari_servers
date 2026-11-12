# ARI-Demo Servers

Network servers of the demos of ARIS: they run on the machine with the GPU (DGX Spark, Jetson, cluster)
and the robot reaches them over HTTP. The clients of these servers stay in `ari_control`.

Architecture: [ari_demo_arch.png](ari_demo_arch.png). This repository is the **DGX-Spark** box: the three HTTP
servers and the rule *no concurrent executions*. Visual-Perception (Jetson), Behavioral-Trees, HTTP clients and
Robot Control (Main-PC) are in `ari_control`.

| Server | Folder | Port (`<..._PORT>`) | State |
|---|---|---|---|
| Whisper (speech to text) | [whisper/](whisper/README.md) | 8000 | model as a parameter, tested on the Spark |
| LLM (+RAG) (answers) | [llm/](llm/README.md) | 8010 | ChatGPT backend (reference, from the actions of Rasa) with OpenAI RAG; local backend and retriever to port |
| RASA (dialogue) | [rasa/](rasa/README.md) | 5002, 5003, 5004 (one per bot) | the 3 bots of PepperDemo and their actions behind the proxy with the lock |

```
ari_servers/
├── config/servers.yaml   ports, models and options of every server, execution lock
├── common/ari_servers/   shared code: config.py, http.py (HTTPService), execution.py (lock), latency_log.py
├── whisper/              whisper_server/ (package), docker/, conda/, tools/
├── llm/                  llm_server/ (package): backends/ (abstract class, ChatGPT, ...), server.py, rag.py; docker/
├── rasa/                 bots/ (social, rock_paper_scissor, connect_four), rasa_proxy/ (package), docker/
├── models/  audio_files/  latency_logs/  run/
```

## Configuration
Every server is a dataclass of options (e.g. `WhisperConfig`), filled from the lowest to the highest priority by:
its defaults, its section of [config/servers.yaml](config/servers.yaml), the environment `<PREFIX>_<FIELD>`
(`WHISPER_MODEL`, `LLM_PORT`, `RASA_ROUTES`...), the command line `--<field>`. The **model** of each server is a
field: change it in the YAML file, or for one run with a variable or an option.
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
take it (`rasa.exclusive: false`) and the LLM server takes it during the generation.

## Run without containers
```bash
export PYTHONPATH=$PWD/common:$PWD/whisper:$PWD/llm:$PWD/rasa
python -m whisper_server        # needs the libraries of whisper (images or conda environment)
python -m llm_server --skip-load
python -m rasa_proxy            # standard library only
```

Shared folders (the workspace of every server, mounted as `/workspace` in the containers):

| Folder | Content |
|---|---|
| `config/` | `servers.yaml`, `rasa.env`; `secrets.env` with the keys, not versioned |
| `models/` | models: `silero/silero_vad.onnx` (versioned), `whisper/` (exported at the first start), `rasa/` (trained bots): not versioned |
| `audio_files/` | the 5 Italian WAV of the tests, their reference transcripts (`references.csv`) and the [report of the Whisper models](audio_files/REPORT_modelli_whisper.md) |
| `latency_logs/` | latency log of the servers (`latency_<date>_<host>.csv`), not versioned |
| `run/` | execution lock, not versioned |

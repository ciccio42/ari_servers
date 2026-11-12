# RASA server

The dialogue of the demo: the three Rasa bots of PepperDemo (`rasa_ws`), the action server of the social bot and
`rasa_proxy`, which makes them follow *no concurrent executions*. The clients keep the ports and the Rasa HTTP API of
today (`/webhooks/rest/webhook`, `/conversations/<id>/tracker/events`, `/model/parse`, `/`): only the address of the
machine changes.

```
 client ──HTTP :5002──► rasa_proxy ──(lock)──► social             :5012 ──► action server :5055 ──► OpenAI
 client ──HTTP :5003──►            ──(lock)──► rock_paper_scissor :5013
 client ──HTTP :5004──►            ──(lock)──► connect_four       :5014
```

| Path | Content |
|---|---|
| `bots/social/` | `rasa_bot` of PepperDemo: `config/` (endpoints, credentials, `models/<it\|en>/` config, domain, NLU data, stories), `actions/` (custom actions) |
| `bots/rock_paper_scissor/`, `bots/connect_four/` | the bots of the games: `config/` only, they have no custom actions |
| `rasa_proxy/` | proxy with the execution lock, one listener per bot (standard library only) |
| `docker/` | image `ari_rasa_server` (CPU, Rasa 3.6.21, the versions of the PepperDemo image), `entrypoint.bash`, `run.bash` |
| `../models/rasa/<bot>/<it\|en>/` | trained models, not versioned |

## Proxy
| Field (`rasa` in servers.yaml, `RASA_<FIELD>`, `--<field>`) | Default | |
|---|---|---|
| `routes` | `5002=http://localhost:5012, 5003=…5013, 5004=…5014` | `<client port>=<bot>`: one listener per route, one lock for all of them |
| `exclusive` | true | take the lock; **false** if a custom action calls another server of the machine (e.g. LLM): with the lock held it would wait forever |
| `timeout` | 60 | seconds for one request to a bot |

`GET /health` on every port: `{"status": "ok", "service": "rasa_proxy_<port>", "upstream", "upstream_ok"}`. Bot not
reachable: `502`. If one listener stops (e.g. port already in use), the proxy stops. Every request writes a
`rasa_proxy.done` line (port, path, `queue_s`, `rasa_s`, `total_s`) in `latency_logs/latency_<date>_<host>.csv`.

The lock is held for the whole answer of a bot, OpenAI calls of the custom actions included: Whisper and the LLM wait
for it (`queue_s`).

## Configuration
Three files, in `config/`:

| File | Versioned | What to change there |
|---|---|---|
| [config/rasa.env](../config/rasa.env) | yes | language, OpenAI model, **RAG on/off** and its parameters, flags of the demo (`KNOWLEDGE_GRAPH_RUNNING`, `MAX_CHAR`, `DEBUG`) |
| `config/secrets.env` | **no** | `OPENAI_API_KEY` only |
| [config/servers.yaml](../config/servers.yaml), section `rasa` | yes | ports of the clients and bots (`routes`), execution lock (`exclusive`), `timeout` |

The key, once:
```bash
echo 'OPENAI_API_KEY=sk-...' > config/secrets.env && chmod 600 config/secrets.env
```
Never write the key in `rasa.env` or in another versioned file.

Every parameter of `rasa.env` is commented with its meaning and its default; a line commented with `#` keeps the
default of the code. The main switches:

| Parameter | Value | |
|---|---|---|
| `DEMO_LANGUAGE` | `Italian` | `Italian` or `English` |
| `OPENAI_MODEL` | `gpt-4o` (the one of the demo; code default `gpt-4o-mini`) | model of the answers |
| `OPENAI_FILE_SEARCH` | `true` | **RAG**: `true` searches the documents of the OpenAI vector store before answering, `false` answers without them (faster) |
| `OPENAI_VECTOR_STORE_ID`, `OPENAI_RAG_*`, `OPENAI_FILE_SEARCH_*` | commented | parameters of the RAG, used only with `OPENAI_FILE_SEARCH=true` |
| `KNOWLEDGE_GRAPH_RUNNING` | `False` | `True`: `action_add_info_kg` uses the knowledge graph |

Format of the files (`docker run --env-file`): `VARIABLE=value`, **no quotes** (they become part of the value:
`run.bash` refuses such a file), no comment after the value. A variable exported in the shell wins over the files,
e.g. `OPENAI_FILE_SEARCH=false bash rasa/docker/run.bash start` for one run without RAG. `RASA_ENV_FILE=<file>`
uses another file instead of `rasa.env` (e.g. one for English); `secrets.env` is read anyway.

## Run with Docker (Spark)
From `ari_servers`:
```bash
bash rasa/docker/run.bash build      # image ari_rasa_server, once
bash rasa/docker/run.bash start      # with config/rasa.env and config/secrets.env; ready in about 15-60 s
bash rasa/docker/run.bash status     # /health of 5002, 5003, 5004
bash rasa/docker/run.bash test "qual è la ricetta della pizza?"   # one message to every bot
bash rasa/docker/run.bash logs
bash rasa/docker/run.bash stop
```
**After a change** of the files of `config/`: `stop`, then `start` (the container reads them only at the
start). A change of the actions or of the bots needs the same; a change of the image (`docker/`) needs `build` too.

The container `rasa_server` runs the action server, the three bots (`rasa run --enable-api`) and the proxy; if one
of them stops, the container stops and Docker restarts it. Stop the Rasa of PepperDemo first (`rasa_dev_container`):
it uses the same ports.

## Models
`rasa run -m <folder>` loads the newest model of the folder. The models copied from PepperDemo are the ones it loaded
on 2026-11-12, one per bot and language. To train them again from `config/models/<lang>`:
```bash
bash rasa/docker/run.bash train                 # all the bots, it and en
bash rasa/docker/run.bash train social it
```

## Changes from PepperDemo
- Bots renamed: `rasa_bot` → `social`, `rock_paper_scissor_bot` → `rock_paper_scissor`, `connect_four_bot` →
  `connect_four`. Internal ports 5012–5014 instead of 5002–5004, which belong to the proxy.
- Actions: only the modules used (`actions.py`, `chat_gpt.py`). The parts of `demo_utils` they need are copied next
  to them (`text_manager.py` with `phrases_rasa.yml`, `pluralize.py`, `runner.py`, `events.py`); the latency log is
  the one of `common/ari_servers`. Not copied: `gpt.py`, `wiki.py`, `chat_gpt_old.py`, `llm_rag.py` (not imported).
- The OpenAI key is read from `OPENAI_API_KEY` (`config/secrets.env`), not written in the code.
- Image: CPU only, without torch and the libraries of the LLM, which belong to the LLM server.

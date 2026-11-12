# RASA server

The dialogue of the demo: the three Rasa bots of PepperDemo (`rasa_ws`), the action server of the social bot and
`rasa_proxy`, which makes them follow *no concurrent executions*. The clients keep the ports and the Rasa HTTP API of
today (`/webhooks/rest/webhook`, `/conversations/<id>/tracker/events`, `/model/parse`, `/`): only the address of the
machine changes.

```
 client ──HTTP :5002──► rasa_proxy ──► social             :5012 ──► action server :5055 ──HTTP──► LLM server :8010
 client ──HTTP :5003──►            ──► rock_paper_scissor :5013                                    (llm/, lock)
 client ──HTTP :5004──►            ──► connect_four       :5014
```

| Path | Content |
|---|---|
| `bots/social/` | `rasa_bot` of PepperDemo: `config/` (endpoints, credentials, `models/<it\|en>/` config, domain, NLU data, stories), `actions/` (custom actions; `llm_client.py`: client of the LLM server) |
| `bots/rock_paper_scissor/`, `bots/connect_four/` | the bots of the games: `config/` only, they have no custom actions |
| `rasa_proxy/` | proxy with the execution lock, one listener per bot (standard library only) |
| `docker/` | image `ari_rasa_server` (CPU, Rasa 3.6.21, the versions of the PepperDemo image), `entrypoint.bash`, `run.bash` |
| `../models/rasa/<bot>/<it\|en>/` | trained models, not versioned |

## Proxy
| Field (`rasa` in servers.yaml, `RASA_<FIELD>`, `--<field>`) | Default | |
|---|---|---|
| `routes` | `5002=http://localhost:5012, 5003=…5013, 5004=…5014` | `<client port>=<bot>`: one listener per route, one lock for all of them |
| `exclusive` | **false** in servers.yaml | take the lock for every request. False because the social action calls the LLM server, which takes the lock while it generates: with the lock held by the proxy it would wait forever |
| `timeout` | 60 | seconds for one request to a bot |

`GET /health` on every port: `{"status": "ok", "service": "rasa_proxy_<port>", "upstream", "upstream_ok"}`. Bot not
reachable: `502`. If one listener stops (e.g. port already in use), the proxy stops. Every request writes a
`rasa_proxy.done` line (port, path, `queue_s`, `rasa_s`, `total_s`) in `latency_logs/latency_<date>_<host>.csv`.

With `exclusive: false` the bots answer without the lock (10–40 ms of CPU); the lock is taken by the LLM server
during the generation of the answers of the social bot.

## Answers of the social bot
`action_social` (general and medical questions), `action_gpt_4o` (questions about the scene, with the image of the
camera) and `action_add_info_kg` (knowledge graph) ask the [LLM server](../llm/README.md) through
`actions/llm_client.py`, with `conversation_id` = the sender of the tracker. Model, RAG, OpenAI key and history are
configured there; the prompts stay here (`phrases_rasa.yml`: "Medical Prompt", "KG_Prompt"). If the LLM server does
not answer, the actions answer with their courtesy phrase (`So1`) and the dialogue goes on.

## Configuration
| File | Versioned | What to change there |
|---|---|---|
| [config/rasa.env](../config/rasa.env) | yes | language, address of the LLM server (`LLM_URL`, `LLM_TIMEOUT`), flags of the demo (`KNOWLEDGE_GRAPH_RUNNING`, `MAX_CHAR`, `DEBUG`) |
| [config/servers.yaml](../config/servers.yaml), section `rasa` | yes | ports of the clients and bots (`routes`), execution lock (`exclusive`), `timeout` |
| [config/servers.yaml](../config/servers.yaml), sections `llm` and `openai`; `config/secrets.env` | yes; **no** | model, **RAG on/off**, OpenAI parameters and key: they belong to the LLM server, see [llm/README.md](../llm/README.md#configuration) |

Every parameter of `rasa.env` is commented with its meaning and its default; a line commented with `#` keeps the
default of the code. Format (`docker run --env-file`): `VARIABLE=value`, **no quotes** (they become part of the
value: `run.bash` refuses such a file), no comment after the value. A variable exported in the shell wins over the
file. `RASA_ENV_FILE=<file>` uses another file instead of `rasa.env` (e.g. one for English).

## Run with Docker (Spark)
From `ari_servers`:
```bash
bash rasa/docker/run.bash build      # image ari_rasa_server, once
bash llm/docker/run.bash start       # the LLM server first: the answers of the social bot
bash rasa/docker/run.bash start      # with config/rasa.env; ready in about 15-60 s
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
- Actions: only `actions.py`, and the parts of `demo_utils` it needs copied next to it (`text_manager.py` with
  `phrases_rasa.yml`, `pluralize.py`, `events.py`). Not copied: `gpt.py`, `wiki.py`, `chat_gpt_old.py`, `llm_rag.py`
  (not imported).
- ChatGPT (`chat_gpt.py`) moved to the LLM server, as its reference backend (`llm/llm_server/backends/openai_chat.py`):
  the actions are its HTTP clients. Every user has their own history (in PepperDemo one history for everybody, and
  the questions about the scene used the last prompt set by another action; now the default prompt of the server).
- Image: CPU only, without openai, torch and the libraries of the LLM, which belong to the LLM server.

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

## Run with Docker (Spark)
From `ari_servers`:
```bash
bash rasa/docker/run.bash build
RASA_ENV_FILE=~/rasa.env bash rasa/docker/run.bash start    # ~/rasa.env: OPENAI_API_KEY=... (not versioned)
bash rasa/docker/run.bash status     # /health of 5002, 5003, 5004
bash rasa/docker/run.bash test       # one message to every bot
bash rasa/docker/run.bash logs
bash rasa/docker/run.bash stop
```
The container `rasa_server` runs the action server, the three bots (`rasa run --enable-api`) and the proxy; if one
of them stops, the container stops and Docker restarts it. Stop the Rasa of PepperDemo first (`rasa_dev_container`):
it uses the same ports.

| Variable | Default | |
|---|---|---|
| `DEMO_LANGUAGE` | `Italian` | `Italian` or `English`: models of `models/rasa/<bot>/<it\|en>` |
| `OPENAI_API_KEY` | none | **needed** by the action server: unset, it stops at the start. Never in the code |
| `OPENAI_MODEL`, `OPENAI_FILE_SEARCH`, `OPENAI_*` | see `bots/social/actions/chat_gpt.py` | ChatGPT of the social bot and its RAG |
| `KNOWLEDGE_GRAPH_RUNNING`, `MAX_CHAR`, `DEBUG` | `False`, 250, `False` | as in PepperDemo |
| `RASA_BOTS` | `social:5012 rock_paper_scissor:5013 connect_four:5014` | `<bot>:<internal port>`, must match `rasa.routes` |
| `ACTIONS_PORT` | 5055 | as `action_endpoint` of `bots/social/config/endpoints.yml` |

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
- The OpenAI key is read from `OPENAI_API_KEY`, not written in the code.
- Image: CPU only, without torch and the libraries of the LLM, which belong to the LLM server.

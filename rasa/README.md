# RASA server

Rasa has its own HTTP API (`rasa run --enable-api`): it is not rewritten. `rasa_proxy` stands on the port of the
architecture (`<RASA_PORT>`, default 5005) in front of Rasa (default 5006) and forwards every request unchanged,
holding the execution lock: Rasa follows *no concurrent executions* too, and the clients of the demo do not change
(`/webhooks/rest/webhook`, `/conversations/<id>/tracker/events`, `/model/parse`, `/`).

```
 client (Main-PC) ──HTTP :5005──► rasa_proxy ──(lock)──► Rasa :5006 ──► action server
```

| Field (`rasa` in servers.yaml, `RASA_<FIELD>`, `--<field>`) | Default | |
|---|---|---|
| `port` | 5005 | port of the clients |
| `upstream` | `http://localhost:5006` | Rasa |
| `exclusive` | true | take the lock; **false** if a custom action calls another server of the machine (e.g. LLM): with the lock held it would wait forever |
| `timeout` | 60 | seconds for one request to Rasa |

`GET /health` of the proxy: `{"status": "ok", "service": "rasa_proxy", "upstream", "upstream_ok"}`. Rasa not
reachable: `502`.

```bash
rasa run --enable-api --port 5006           # in the Rasa project (PepperDemo/rasa_ws/rasa_bot)
export PYTHONPATH=$PWD/common:$PWD/rasa     # from ari_servers
python -m rasa_proxy                        # standard library only
```

Tested on 2026-11-12 with a fake Rasa: requests forwarded unchanged, two parallel webhooks served one after the other
(0.30 and 0.60 s), 502 with Rasa down. To do: Rasa itself (image or environment of the Rasa project) and its start
together with the proxy.

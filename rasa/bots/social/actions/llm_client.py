"""Client of the LLM server (llm/ of ari_servers): the custom actions ask it the answers, instead of calling
ChatGPT themselves. Standard library only.

LLM_URL: address of the LLM server (default http://localhost:8010, the same machine). LLM_TIMEOUT: seconds to wait
for one answer; a little more than llm.timeout of the server, so its error arrives before this timeout.
Every failure (server unreachable, backend error, timeout) gives "": the actions then answer with their courtesy
phrase, as they did with ChatGPT.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8010").rstrip("/")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "70"))

log = logging.getLogger(__name__)


def _post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(f"{LLM_URL}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
        return json.load(response)


def chat(conversation_id: str, message: str, system: Optional[str] = None, rag: Optional[bool] = None,
         image_base64: Optional[str] = None) -> str:
    """Answer of the LLM server to `message`, in the conversation `conversation_id` (it keeps the history).
    system: system prompt of this request (default: the one of the server). rag: search the documents
    (default: llm.rag of the server). image_base64: a JPEG, for the questions about the scene."""
    payload = {"conversation_id": conversation_id, "message": message}
    for key, value in (("system", system), ("rag", rag), ("image_base64", image_base64)):
        if value is not None:
            payload[key] = value
    start = time.perf_counter()
    try:
        reply = _post("/chat", payload)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        log.error("LLM server %s: %s %s", LLM_URL, e.code, detail)
        return ""
    except (OSError, ValueError) as e:  # unreachable, timeout, bad JSON
        log.error("LLM server %s not reachable: %s", LLM_URL, e)
        return ""
    log.info("LLM server: %s, rag %s, history %s, %.2f s (queue %.2f s)", reply.get("served_model") or reply.get("model"),
             reply.get("rag"), reply.get("history"), time.perf_counter() - start, reply.get("queue_s", 0))
    return reply.get("text", "")


def reset(conversation_id: str) -> None:
    try:
        _post("/reset", {"conversation_id": conversation_id})
    except (OSError, ValueError) as e:
        log.error("LLM server %s: reset failed: %s", LLM_URL, e)

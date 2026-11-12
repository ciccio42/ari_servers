"""ChatGPT backend: the reference implementation of ChatBackend, ported from the class ChatGpt of the custom actions
of Rasa (PepperDemo rasa_ws/rasa_bot/actions/chat_gpt.py).

- text and vision: Chat Completions API;
- RAG (rag=True): Responses API with the file_search tool on the OpenAI vector store of the diabetology and
  cardiology documents;
- one new attempt after a server or connection error, within the timeout of the request;
- the citations written inline by file_search are removed from the answer.

The model is LLMConfig.model (llm.model). The other parameters are in the section `openai` of config/servers.yaml,
or OPENAI_<FIELD> (the variables of the actions of Rasa: OPENAI_VECTOR_STORE_ID, OPENAI_RAG_MODEL, ...).
The key only from the environment: OPENAI_API_KEY (config/secrets.env), never in a file of the repository.
"""
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

from ari_servers import config
from ari_servers.latency_log import mark

from .base import BackendError, ChatBackend, Message, Reply

log = logging.getLogger("openai")

EFFORTS = ("minimal", "low", "medium", "high")
# only these models accept the reasoning parameters: gpt-4o and gpt-4o-mini reject them
REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")
FILE_SEARCH_HINT = (
    "\n\nDOCUMENTI DI RIFERIMENTO: per le domande di diabetologia e cardiologia (terapia, valori di riferimento, "
    "diagnosi, emoglobina) cerca prima nei documenti con lo strumento file_search e basa la risposta su di essi. "
    "Non citare i documenti, i nomi dei file o lo strumento nella risposta."
)


@dataclass
class OpenAIConfig:
    # gpt-5 models reason before answering; "low" cuts the hidden reasoning tokens and the latency (37 s -> 8 s with
    # file_search). Only for the reasoning models
    reasoning_effort: str = "low"
    # --- RAG (file_search), used with rag=True
    vector_store_id: str = "vs_6adf80ee4cf481919b35fecdd7d1efd9"  # diabetology and cardiology sources
    rag_model: Optional[str] = None  # model of the RAG answers; none: llm.model
    rag_reasoning_effort: Optional[str] = None  # none: reasoning_effort
    file_search_max_results: int = 5  # passages of every search (1-50): more passages, more tokens and latency
    # "required": always search before answering; with "auto" the model searched for only 8 of the 22 test questions
    file_search_tool_choice: str = "required"
    file_search_score_threshold: Optional[float] = None  # passages below this relevance (0-1) are discarded
    file_search_ranker: Optional[str] = None  # "auto" or a named ranker, e.g. "default-2024-11-15"
    rag_max_output_tokens: Optional[int] = None  # generated tokens, reasoning included; none: no limit
    rag_temperature: Optional[float] = None  # 0-2, only for the non-reasoning models; none: OpenAI default (1)
    rag_verbosity: Optional[str] = None  # answer length of the gpt-5 models: low, medium, high
    # --- attempts
    max_retries: int = 1  # new attempts after a server error (5xx) or a connection error/timeout
    min_retry_time: float = 10.0  # seconds: no new attempt if less than this is left of the timeout

    def check(self):
        if self.reasoning_effort not in EFFORTS or (self.rag_reasoning_effort or "low") not in EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {EFFORTS}")
        if self.file_search_tool_choice not in ("required", "auto"):
            raise ValueError("file_search_tool_choice must be required or auto")
        if not 1 <= self.file_search_max_results <= 50:
            raise ValueError("file_search_max_results must be between 1 and 50")
        if self.rag_verbosity not in (None, "low", "medium", "high"):
            raise ValueError("rag_verbosity must be low, medium or high")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODELS)


class ChatGPTBackend(ChatBackend):
    name = "openai"
    native_rag = True
    vision = True

    # patterns removed from the answers
    _BACK_PATTERN = re.compile(r"【.*?】")
    _NEW_LINE_PATTERN = re.compile(r"\\n\\n\d+")
    # file_search citations written inline by the model, between the private use characters U+E200 and U+E201
    # (shown as "fileciteturn0file2turn0file3")
    _CITE_PATTERN = re.compile("\\s*\ue200[^\ue201]*\ue201")

    def __init__(self, cfg):
        super().__init__(cfg)
        self.openai = config.load(OpenAIConfig, "openai", "OPENAI", argv=[])
        self.openai.check()
        self._client = None

    @property
    def rag_model(self) -> str:
        return self.openai.rag_model or self.cfg.model

    def load(self) -> None:
        from openai import OpenAI  # only for this backend

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set: write it in config/secrets.env")
        # no automatic retries of the library: generate() retries within the timeout of the request
        self._client = OpenAI(api_key=key, max_retries=0)
        log.info("ChatGPT model %s, temperature %s; RAG: %s", self.cfg.model,
                 None if is_reasoning_model(self.cfg.model) else self.cfg.temperature, self.rag_settings())

    def describe(self) -> dict:
        return {**super().describe(), "rag_model": self.rag_model, "vector_store_id": self.openai.vector_store_id}

    def rag_settings(self) -> dict:
        return {**asdict(self.openai), "rag_model": self.rag_model}

    # --- requests

    def _rag_kwargs(self) -> dict:
        """Parameters of a Responses API call with file_search."""
        o, model = self.openai, self.rag_model
        tool = {"type": "file_search", "vector_store_ids": [o.vector_store_id], "max_num_results": o.file_search_max_results}
        ranking = {k: v for k, v in (("ranker", o.file_search_ranker), ("score_threshold", o.file_search_score_threshold))
                   if v is not None}
        if ranking:
            tool["ranking_options"] = ranking
        kwargs = {"model": model, "tools": [tool], "tool_choice": o.file_search_tool_choice}
        if o.rag_max_output_tokens is not None:
            kwargs["max_output_tokens"] = o.rag_max_output_tokens
        if is_reasoning_model(model):
            kwargs["reasoning"] = {"effort": o.rag_reasoning_effort or o.reasoning_effort}
            if o.rag_verbosity is not None and model.startswith("gpt-5"):
                kwargs["text"] = {"verbosity": o.rag_verbosity}
        elif o.rag_temperature is not None:
            kwargs["temperature"] = o.rag_temperature
        return kwargs

    @staticmethod
    def _to_responses_input(message: Message) -> Message:
        """A Chat Completions message in the format of the Responses API (the image parts differ)."""
        if isinstance(message["content"], str):
            return message
        parts = []
        for part in message["content"]:
            if part["type"] == "text":
                parts.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image_url":
                parts.append({"type": "input_image", "image_url": part["image_url"]["url"]})
        return {"role": message["role"], "content": parts}

    def _request(self, messages: List[Message], rag: bool, temperature: float, timeout: float):
        """One call to OpenAI: (text, served model)."""
        if rag:
            system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            response = self._client.responses.create(
                instructions=system + FILE_SEARCH_HINT,
                input=[self._to_responses_input(m) for m in messages if m["role"] != "system"],
                timeout=timeout,
                **self._rag_kwargs(),
            )
            return response.output_text, getattr(response, "model", None)
        if is_reasoning_model(self.cfg.model):
            extra = {"reasoning_effort": self.openai.reasoning_effort}
        else:
            extra = {"temperature": temperature} if temperature is not None else {}
        response = self._client.chat.completions.create(model=self.cfg.model, messages=messages, timeout=timeout,
                                                        **extra)
        return response.choices[0].message.content, getattr(response, "model", None)

    def _filter_text(self, text: str) -> str:
        text = self._BACK_PATTERN.sub("", text)
        text = self._CITE_PATTERN.sub("", text)
        return self._NEW_LINE_PATTERN.sub("", text)

    def generate(self, messages: List[Message], *, rag: bool, temperature: float, max_new_tokens: int,
                 timeout: float) -> Reply:
        from openai import APIConnectionError, InternalServerError

        model = self.rag_model if rag else self.cfg.model
        start = time.time()
        mark("openai", "request", model=model, file_search=rag)
        reply, served, error = "", None, "no answer"
        for attempt in range(1 + self.openai.max_retries):
            # every attempt uses what is left of the timeout, so the client still gets an answer in time
            remaining = timeout - (time.time() - start)
            try:
                reply, served = self._request(messages, rag, temperature, remaining)
                break
            except (InternalServerError, APIConnectionError) as e:  # APIConnectionError includes the timeouts
                error = str(e)
                log.error("OpenAI request failed (attempt %d): %s", attempt + 1, e)
                if timeout - (time.time() - start) < self.openai.min_retry_time:
                    break
            except Exception as e:  # authentication, bad request...: a new attempt would fail too
                error = str(e)
                log.error("OpenAI request failed: %s", e)
                break
        mark("openai", "reply", model=model, served_model=served, file_search=rag, chars=len(reply or ""),
             attempts=attempt + 1, elapsed_s=round(time.time() - start, 3))
        if not reply:
            raise BackendError(f"OpenAI: {error}")
        return Reply(text=self._filter_text(reply), model=model, served_model=served, rag=rag, attempts=attempt + 1)

import os
import re
import logging
import time

from openai import APIConnectionError, InternalServerError, OpenAI

from ari_servers.latency_log import mark

from .text_manager import TextManagerSocial, MAX_CHAR

# ---------------------------------------------------------------------------
# Configuration – all sensitive values come from environment variables
# ---------------------------------------------------------------------------
# never in the code: environment of the action server. Unset: OpenAI() stops the action server at the start
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or None

def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name}={value!r}: allowed values are {', '.join(choices)}")
    return value


def _env_number(name: str, cast: type, default=None):
    """Optional numeric variable: unset or empty -> default (None means 'do not send the parameter')."""
    value = os.environ.get(name, "").strip()
    return cast(value) if value else default


OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# gpt-5 models reason before answering; "low" cuts the hidden reasoning tokens and the latency (37 s -> 8 s with file_search)
REASONING_EFFORT: str = _env_choice("OPENAI_REASONING_EFFORT", "low", ("minimal", "low", "medium", "high"))
# only these models accept the reasoning parameters: gpt-4o and gpt-4o-mini reject them
REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")
# sampling temperature (0-2) of the answers without file_search (Chat Completions), only for the non-reasoning models:
# low = the same question gets the same kind of answer (with the OpenAI default 1 the Unisa question was refused 9 times
# out of 10 and answered other times); empty = OpenAI default
TEMPERATURE = _env_number("OPENAI_TEMPERATURE", float, 0.3)

# ---------------------------------------------------------------------------
# RAG with ChatGPT: Responses API + file_search on the OpenAI vector store (medical questions, ActionSocial)
# Every parameter can be overridden with the environment variable next to it.
# ---------------------------------------------------------------------------
# False: ActionSocial answers with Chat Completions and the general knowledge of OPENAI_MODEL, without searching
# the vector store (faster: no Responses API and no search); the OPENAI_RAG_* / FILE_SEARCH_* settings are then unused
FILE_SEARCH_ENABLED: bool = _env_choice("OPENAI_FILE_SEARCH", "true", ("true", "false")) == "true"
# model used only for the medical questions; by default the same as OPENAI_MODEL
RAG_MODEL: str = os.environ.get("OPENAI_RAG_MODEL", OPENAI_MODEL)
# reasoning effort of RAG_MODEL (only for the reasoning models)
RAG_REASONING_EFFORT: str = _env_choice("OPENAI_RAG_REASONING_EFFORT", REASONING_EFFORT, ("minimal", "low", "medium", "high"))
# OpenAI vector store with the diabetology and cardiology sources (searched with the file_search tool)
VECTOR_STORE_ID: str = os.environ.get("OPENAI_VECTOR_STORE_ID", "vs_6adf80ee4cf481919b35fecdd7d1efd9")
# passages returned by every search (1-50): more passages, more input tokens and latency
FILE_SEARCH_MAX_RESULTS: int = _env_number("OPENAI_FILE_SEARCH_MAX_RESULTS", int, 5)
# "required": always search before answering; with "auto" the model searched for only 8 of the 22 test questions
FILE_SEARCH_TOOL_CHOICE: str = _env_choice("OPENAI_FILE_SEARCH_TOOL_CHOICE", "required", ("required", "auto"))
# passages with a relevance score below this (0-1) are discarded; unset = OpenAI default (no threshold)
FILE_SEARCH_SCORE_THRESHOLD = _env_number("OPENAI_FILE_SEARCH_SCORE_THRESHOLD", float)
# ranker of the search results: "auto" or a named one such as "default-2024-11-15"; unset = OpenAI default
FILE_SEARCH_RANKER = os.environ.get("OPENAI_FILE_SEARCH_RANKER", "").strip() or None
# limit of generated tokens, reasoning tokens included; unset = no limit (the prompt already asks for MAX_CHAR characters,
# set with the MAX_CHAR environment variable: text_manager.py)
RAG_MAX_OUTPUT_TOKENS = _env_number("OPENAI_RAG_MAX_OUTPUT_TOKENS", int)
# sampling temperature (0-2), only for the non-reasoning models (gpt-4o, gpt-4o-mini); unset = OpenAI default (1)
RAG_TEMPERATURE = _env_number("OPENAI_RAG_TEMPERATURE", float)
# answer length of the gpt-5 models: "low", "medium", "high"; unset = OpenAI default (medium)
RAG_VERBOSITY = os.environ.get("OPENAI_RAG_VERBOSITY", "").strip().lower() or None
FILE_SEARCH_HINT = (
    "\n\nDOCUMENTI DI RIFERIMENTO: per le domande di diabetologia e cardiologia (terapia, valori di riferimento, "
    "diagnosi, emoglobina) cerca prima nei documenti con lo strumento file_search e basa la risposta su di essi. "
    "Non citare i documenti, i nomi dei file o lo strumento nella risposta."
)

assert 1 <= FILE_SEARCH_MAX_RESULTS <= 50, "OPENAI_FILE_SEARCH_MAX_RESULTS must be between 1 and 50"
assert RAG_VERBOSITY in (None, "low", "medium", "high"), "OPENAI_RAG_VERBOSITY must be low, medium or high"


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODELS)


def rag_request_kwargs(model: str = RAG_MODEL, effort: str = RAG_REASONING_EFFORT,
                       tool_choice: str = FILE_SEARCH_TOOL_CHOICE) -> dict:
    """Parameters of a Responses API call with file_search, from the RAG configuration above.
    Shared by ChatGpt and the comparison script (rag_medico/confronto_locale_gpt.py)."""
    tool = {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID], "max_num_results": FILE_SEARCH_MAX_RESULTS}
    ranking = {k: v for k, v in (("ranker", FILE_SEARCH_RANKER), ("score_threshold", FILE_SEARCH_SCORE_THRESHOLD))
               if v is not None}
    if ranking:
        tool["ranking_options"] = ranking
    kwargs = {"model": model, "tools": [tool], "tool_choice": tool_choice}
    if RAG_MAX_OUTPUT_TOKENS is not None:
        kwargs["max_output_tokens"] = RAG_MAX_OUTPUT_TOKENS
    if is_reasoning_model(model):
        kwargs["reasoning"] = {"effort": effort}
        if RAG_VERBOSITY is not None and model.startswith("gpt-5"):
            kwargs["text"] = {"verbosity": RAG_VERBOSITY}
    elif RAG_TEMPERATURE is not None:
        kwargs["temperature"] = RAG_TEMPERATURE
    return kwargs


def rag_config() -> dict:
    """The RAG parameters in use, for the logs and the comparison results."""
    return {
        "file_search": FILE_SEARCH_ENABLED,
        "modello": RAG_MODEL,
        "reasoning_effort": RAG_REASONING_EFFORT if is_reasoning_model(RAG_MODEL) else None,
        "vector_store": VECTOR_STORE_ID, "max_num_results": FILE_SEARCH_MAX_RESULTS,
        "tool_choice": FILE_SEARCH_TOOL_CHOICE, "score_threshold": FILE_SEARCH_SCORE_THRESHOLD,
        "ranker": FILE_SEARCH_RANKER, "max_output_tokens": RAG_MAX_OUTPUT_TOKENS,
        "temperature": RAG_TEMPERATURE if not is_reasoning_model(RAG_MODEL) else None,
        "verbosity": RAG_VERBOSITY if RAG_MODEL.startswith("gpt-5") else None,
    }

# DEBUG=True enables the debug writes on file (prompt, response time)
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
MAX_HISTORY_LENGTH: int = 4  # Must be even – e.g. 4 = 2 Q + 2 A turns
TIMEOUT_RESPONSE: int = 60   # seconds, for all the attempts together
MAX_RETRIES: int = 1         # new attempts after a server error (5xx) or a connection error/timeout
MIN_RETRY_TIME: int = 10     # seconds: no new attempt if less than this is left of TIMEOUT_RESPONSE

assert MAX_HISTORY_LENGTH % 2 == 0, "MAX_HISTORY_LENGTH must be an even number"


class ChatGpt:
    """Wrapper around the OpenAI Chat Completions API (text + vision)."""

    PROMPT_SYSTEM = (
        "You are Ari, a social robot programmed to answer specific questions and general knowledge queries. "
        "To help you, you will be provided with information about the people interacting with you, the setting of the conversation, "
        "and the relationships between the characters in the scene. Tailor your response to the context of the environment. "
        "Formulate each of your responses using language that is as natural and conversational as possible, so that you come across "
        "as very sociable. Avoid being overly didactic and use the provided information only when necessary. "
        "Keep your responses within {max_char} characters and be as natural as possible.\n\n"

        "ROBOT QUESTIONS: If the user asks who you are, what your name is, or what you are, reply exactly: "
        "'I am Ari, the intelligent robot of the University of Salerno.' "
        "If the user asks what you can do, what your capabilities are, or how you can help, reply exactly: "
        "'I can recognize objects in the environment. I can also recognize your gender, age and emotions, and I can answer general-knowledge questions.' "
        "Do not add anything else for these questions.\n\n"

        "KNOWLEDGE GRAPH: If a JSON object containing information about people, preferences, the environment or relationships is "
        "provided with the user's message, use it as the primary source for questions about a person, their preferences, "
        "or the current scene. Use only the information relevant to the question. Do not mention the JSON, the knowledge graph, "
        "or its field names in your answer. If the JSON does not contain information relevant to the question, answer using your "
        "general knowledge without referring to missing information."
    )

    KG_SYSTEM= (
        "Estrai una tupla (chiave,valore) dal testo: chiave=caratteristica principale \
        (cibo-[tipologia],attività-[tipologia],preferenza-[tipologia],hobby,allergia), valore=contenuto associato. \
        Il campo [tipologia] va sostituito con gradito o sgradito sulla base del contesto della frase se positivo o negativo. \
        Fornisci prima una breve motivazione, poi la tupla. Formato: Motivazione: ... Output: (chiave,valore). La motivazione deve venire prima."
    )
        
        

    # Regex patterns used to clean model output
    _BACK_PATTERN = re.compile(r"【.*?】")
    _NEW_LINE_PATTERN = re.compile(r"\\n\\n\d+")
    # file_search citations written inline by the model: "fileciteturn0file2turn0file3"
    _CITE_PATTERN = re.compile(r"\s*[^]*")

    # ---------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------

    def __init__(self, model: str = OPENAI_MODEL) -> None:
        self._model = model
        self._reasoning = is_reasoning_model(model)
        # No automatic retries: a retry would exceed the caller's wait time anyway
        self._client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        self.TextManager = TextManagerSocial()

        # Callers wait a little longer than the HTTP timeout so they get the reply (or the error) instead of a TimeoutError
        self.TIMEOUT_RESPONSE = TIMEOUT_RESPONSE + 5 #s
        # Full conversation history (text + vision turns share the same window)
        self._history: list[dict] = []

        self._prompt_system = ChatGpt.PROMPT_SYSTEM.replace("{max_char}", str(MAX_CHAR))
        # model that generated the last reply, as returned by OpenAI (dated version, e.g. gpt-4o-2024-08-06)
        self._served_model = None
        logging.info("ChatGPT model: %s; temperature: %s; RAG configuration: %s", self._model,
                     None if self._reasoning else TEMPERATURE, rag_config())

    # ---------------------------------------------------------------------------
    # Prompt management
    # ---------------------------------------------------------------------------

    def set_prompt(self, prompt: str) -> None:
        """Replace the class-level system prompt with one loaded from TextManager."""
        #ChatGpt.PROMPT_SYSTEM = self._text_manager.get_phrase("Prompt").format(MAX_CHAR)
        if DEBUG:
            with open('set_prompt.txt', 'w') as f:
                f.write(f"{prompt}")
        self._prompt_system = prompt

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _filter_text(self, text: str) -> str:
        """Remove unwanted patterns from a model response."""
        text = self._BACK_PATTERN.sub("", text)
        text = self._CITE_PATTERN.sub("", text)
        text = self._NEW_LINE_PATTERN.sub("", text)
        return text

    def _trim_history(self) -> None:
        """Keep the conversation window within MAX_HISTORY_LENGTH turns."""
        while len(self._history) > MAX_HISTORY_LENGTH:
            self._history.pop(0)

    @staticmethod
    def _to_responses_input(message: dict) -> dict:
        """Converts a Chat Completions message into the Responses API format (vision parts differ)."""
        if isinstance(message["content"], str):
            return message
        parts = []
        for part in message["content"]:
            if part["type"] == "text":
                parts.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image_url":
                parts.append({"type": "input_image", "image_url": part["image_url"]["url"]})
        return {"role": message["role"], "content": parts}

    def _request(self, messages: list[dict], file_search: bool, timeout: float) -> str:
        """One call to OpenAI: Responses API with file_search, Chat Completions otherwise."""
        if file_search:
            # model, file_search and generation parameters: RAG configuration at the top of this file
            response = self._client.responses.create(
                instructions=self._prompt_system + FILE_SEARCH_HINT,
                input=[self._to_responses_input(m) for m in messages[1:]],
                timeout=timeout,
                **rag_request_kwargs(),
            )
            self._served_model = getattr(response, "model", None)
            return response.output_text
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            timeout=timeout,
            **({"reasoning_effort": REASONING_EFFORT} if self._reasoning
               else {"temperature": TEMPERATURE} if TEMPERATURE is not None else {}),
        )
        self._served_model = getattr(response, "model", None)
        return response.choices[0].message.content

    def _infer(self, user_message: dict, file_search: bool = False) -> str:
        """
        Core inference call shared by both text and vision paths.

        *user_message* is a fully-formed OpenAI message dict (role + content).
        The system prompt is always injected as the first message.
        With *file_search* the request goes through the Responses API, which searches VECTOR_STORE_ID.
        """
        self._history.append(user_message)
        self._trim_history()

        messages = [
            {"role": "system", "content": self._prompt_system},
            *self._history,
        ]


        start_time = time.time()
        model = RAG_MODEL if file_search else self._model
        mark("openai", "request", model=model, file_search=file_search, max_char=MAX_CHAR)
        self._served_model = None
        reply = ""
        for attempt in range(1 + MAX_RETRIES):
            # every attempt uses what is left of TIMEOUT_RESPONSE, so the caller still gets an answer in time
            remaining = TIMEOUT_RESPONSE - (time.time() - start_time)
            try:
                reply = self._request(messages, file_search, remaining)
                break
            except (InternalServerError, APIConnectionError) as exc:  # APIConnectionError includes timeouts
                logging.error("OpenAI request failed (attempt %d): %s", attempt + 1, exc)
                if TIMEOUT_RESPONSE - (time.time() - start_time) < MIN_RETRY_TIME:
                    break
            except Exception as exc:
                logging.error("OpenAI request failed: %s", exc)
                break

        mark("openai", "reply", model=model, served_model=self._served_model, file_search=file_search,
             chars=len(reply or ""), attempts=attempt + 1, elapsed_s=round(time.time() - start_time, 3))
        logging.info("OpenAI reply: requested model %s, served by %s", model, self._served_model)
        if not reply:
            # the caller answers with its courtesy phrase ("So1") when the reply is empty
            self._history.pop()  # roll back the failed turn
            return ""

        if DEBUG:
            with open("response_time.txt", "w") as f:
                f.write(f"GPT answered in {(time.time() - start_time)} seconds")

        self._history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def reset_conversation(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    async def chat_gpt(self, request: str, file_search: bool = False) -> str:
        """Send a plain-text *request* and return the model reply.
        With *file_search* the model also searches the medical documents of VECTOR_STORE_ID."""

        user_message = {"role": "user", "content": request}
        #reply = "Capo sto rispondendo con la funzione chat_gpt "

        answer = self._infer(user_message, file_search=file_search)
        reply = self._filter_text(answer)
        logging.info("Response (text): %s", reply)
        return reply

    def gpt_4o(self, request: str, image_base64: str) -> str:
        """
        Send *request* together with a base-64-encoded JPEG and return the
        model reply.  The image turn is stored in history like any other turn.
        """
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": request},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
        #reply = "Capo sto rispondendo con la funzione gpt_4o " 
        answer = self._infer(user_message)
        reply = self._filter_text(answer)
         
        logging.info("Response (vision): %s", reply)
        return reply


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import asyncio

    chat = ChatGpt()
    async def main():
        chat.set_prompt(ChatGpt.KG_SYSTEM)
        print(f"KG System Prompt: {chat._prompt_system}")
        print(await chat.chat_gpt("Mi piace la pizza"))
        print(await chat.chat_gpt("Non mi piace il baccalà"))
        print(await chat.chat_gpt("Mi piacciono i libri di Asimov"))
        print(await chat.chat_gpt("Non mi piace il film di Avatar"))
        print(await chat.chat_gpt("Sono allergico alle noci"))
        

        chat.set_prompt(ChatGpt.PROMPT_SYSTEM.replace("{max_char}", str(MAX_CHAR)))
        print(await chat.chat_gpt("Qual è la capitale d'Italia?"))
        

    asyncio.run(main())

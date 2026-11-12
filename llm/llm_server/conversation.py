"""History of the conversations of the LLM server, one per conversation_id (for Rasa: the sender of the tracker).

As the history of ChatGpt in the actions of Rasa: a window of the last `history_length` messages, the current
request included; a request without answer leaves the history as it was. In memory: a restart of the server
starts new conversations. The oldest conversations are dropped beyond `max_conversations`.
"""
import threading
from collections import OrderedDict
from typing import List

from .backends import Message


class Conversations:

    def __init__(self, history_length: int, max_conversations: int):
        if history_length < 1 or history_length % 2:
            raise ValueError("history_length must be an even number, e.g. 4 = 2 requests + 2 answers")
        self.history_length, self.max_conversations = history_length, max_conversations
        self._history: "OrderedDict[str, List[Message]]" = OrderedDict()
        self._lock = threading.Lock()

    def pending(self, conversation_id: str, request: Message) -> List[Message]:
        """History of the conversation with `request` at the end, within the window. Not stored until commit."""
        with self._lock:
            history = list(self._history.get(conversation_id, []))
        return (history + [request])[-self.history_length:]

    def commit(self, conversation_id: str, pending: List[Message], answer: str) -> None:
        with self._lock:
            self._history[conversation_id] = (pending + [{"role": "assistant", "content": answer}])[-self.history_length:]
            self._history.move_to_end(conversation_id)
            while len(self._history) > self.max_conversations:
                self._history.popitem(last=False)

    def reset(self, conversation_id: str) -> bool:
        with self._lock:
            return self._history.pop(conversation_id, None) is not None

    def __len__(self):
        return len(self._history)

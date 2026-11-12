from enum import Enum

class NLPEvent(Enum):
    """
    Enum class including the events that can be raisen by Rasa Server during the analysis of user request.
    """

    DOA = "/doa"
    ROCK_PAPER_SCISSOR = "rock_paper_scissor"
    CONNECT_FOUR = "connect_four"
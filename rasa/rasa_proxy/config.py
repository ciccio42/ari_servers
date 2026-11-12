"""Configuration of the RASA proxy: section `rasa` of config/servers.yaml, RASA_<FIELD>, --<field>."""
from dataclasses import dataclass

from ari_servers import config


@dataclass
class RasaProxyConfig:
    host: str = "0.0.0.0"
    port: int = 5005  # the port the clients of the demo use for Rasa
    upstream: str = "http://localhost:5006"  # Rasa: rasa run --enable-api --port 5006
    timeout: float = 60.0
    # take the execution lock for every request. False if a custom action of Rasa calls another server of the
    # machine (e.g. the LLM server) while Rasa answers: holding the lock, the proxy would make that call wait forever
    exclusive: bool = True
    max_body_mb: int = 16  # the tracker events can carry an image (base64)


def load(argv=None) -> RasaProxyConfig:
    return config.load(RasaProxyConfig, "rasa", "RASA", argv, description="RASA proxy with exclusive execution")

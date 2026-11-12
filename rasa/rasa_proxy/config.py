"""Configuration of the RASA proxy: section `rasa` of config/servers.yaml, RASA_<FIELD>, --<field>."""
from dataclasses import dataclass

from ari_servers import config


@dataclass
class RasaProxyConfig:
    host: str = "0.0.0.0"
    # <port of the clients>=<Rasa bot>, comma separated: the clients of the demo keep their ports, one per bot
    # (social 5002, rock paper scissor 5003, connect four 5004), the bots listen on the internal ones
    routes: str = "5002=http://localhost:5012,5003=http://localhost:5013,5004=http://localhost:5014"
    timeout: float = 60.0
    # take the execution lock for every request. False if a custom action of Rasa calls another server of the
    # machine (e.g. the LLM server) while Rasa answers: holding the lock, the proxy would make that call wait forever
    exclusive: bool = True
    max_body_mb: int = 16  # the tracker events can carry an image (base64)

    def parsed_routes(self) -> list[tuple[int, str]]:
        """[(port, upstream)] of `routes`."""
        result = []
        for item in filter(None, (part.strip() for part in self.routes.split(","))):
            port, sep, upstream = item.partition("=")
            if not sep or not port.strip().isdigit():
                raise SystemExit(f"rasa.routes: expected <port>=http://host:port, got {item!r}")
            result.append((int(port), upstream.strip()))
        if not result:
            raise SystemExit("rasa.routes is empty")
        return result


def load(argv=None) -> RasaProxyConfig:
    return config.load(RasaProxyConfig, "rasa", "RASA", argv, description="RASA proxy with exclusive execution")

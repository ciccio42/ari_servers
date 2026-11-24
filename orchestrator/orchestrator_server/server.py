"""Orchestrator of the models of the DGX Spark (no ROS): the clients ask it for a pair LLM + RAG, it loads that pair
and unloads the previous one, so only one pair at a time takes the memory of the GPU.

A pair is an LLM, a RAG or both:
  llm  "<backend>:<model>" of the LLM server: "ollama:qwen3.6:35b-a3b" (loaded in Ollama), "openai:gpt-4o" (nothing to
       load: the model runs at OpenAI)
  rag  the name of a RAG server (rag/<name>): its container is started; without "llm" the pair uses the generator of
       the RAG (its /health), which is loaded in Ollama with the options of the RAG (num_ctx)
Loading a pair: stop the RAG containers of the other RAGs, unload the other models of Ollama, start the RAG container
and wait for its /health, load the model in Ollama (keep_alive). Whisper, TTS, RASA and the LLM server are not
touched. After the load the clients use the pair through the LLM server: POST /chat with "rag": "<rag>", or with
"llm": "<llm>" for a pair without RAG.

  GET  /health   {"status": "ok", "service": "orchestrator", "state", "active"}
  GET  /status   {"state": "idle|loading|ready|failed", "active": {"llm", "rag"}, "requested", "error", "load_s",
                  "ollama": [models in memory], "rags": {name: {"running", "ready"}}, "llm_server": ok}
  POST /load     {"llm": "...", "rag": "...", "wait": true}: loads the pair. wait=false replies at once (state
                 loading, then GET /status). The same pair already ready: nothing to do. One load at a time: a
                 second request waits for the first
  POST /unload   {}: stops every RAG container and unloads every model of Ollama
Errors: 400 bad request (unknown RAG, backend...), 409 a load is running and wait=false, 502 the load failed.

The containers are started and stopped with the run.bash of the servers (rag/docker/run.bash,
ollama/docker/run.bash): in Docker the orchestrator has the socket of Docker and ari_servers at its path of the host.
Run: python -m orchestrator_server [--port ...] (PYTHONPATH: common and orchestrator).
"""
import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request

from ari_servers.config import WORKSPACE, read_config_file
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import config

log = logging.getLogger("orchestrator")


class LoadError(Exception):
    """The pair could not be loaded: the server replies 502."""


def http_json(url: str, body: dict = None, timeout: float = 5.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read() or b"{}")


class Orchestrator:

    def __init__(self, cfg: config.OrchestratorConfig):
        self.cfg = cfg
        self.root = WORKSPACE  # ari_servers, at the path of the host (the run.bash mount it in the containers)
        self.state, self.active, self.requested, self.error, self.load_s = "idle", {"llm": "", "rag": ""}, None, "", None
        self._load_lock = threading.Lock()  # one load at a time
        self._state_lock = threading.Lock()

    # --- servers.yaml, scripts, containers

    def rag_names(self) -> list:
        return sorted(p.name for p in (self.root / "rag").iterdir() if (p / "environment.yml").is_file())

    def rag_port(self, name: str) -> int:
        return int((read_config_file().get(f"rag_{name}") or {}).get("port", 8030))

    def script(self, *args, timeout: float = 120.0) -> str:
        """Runs a run.bash of ari_servers (path relative to ari_servers): its output, LoadError if it fails."""
        command = ["bash", str(self.root / args[0]), *args[1:]]
        try:
            done = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise LoadError(f"{' '.join(args)}: no end after {timeout:.0f} s")
        if done.returncode != 0:
            raise LoadError(f"{' '.join(args)} failed: {(done.stderr or done.stdout).strip()[-400:]}")
        return done.stdout

    @staticmethod
    def container_running(name: str) -> bool:
        done = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True)
        return done.returncode == 0 and done.stdout.strip() == "true"

    def rag_health(self, name: str):
        try:
            return http_json(f"http://localhost:{self.rag_port(name)}/health", timeout=3)
        except (OSError, ValueError):
            return None

    def ollama_models(self):
        """Models in memory in Ollama, None if it does not answer."""
        try:
            return [m["name"] for m in http_json(f"{self.cfg.ollama_url}/api/ps").get("models", [])]
        except (OSError, ValueError, KeyError):
            return None

    # --- steps of a load

    def stop_rags(self, keep: str = ""):
        for name in self.rag_names():
            if name != keep and self.container_running(f"rag_{name}"):
                log.info("stopping rag_%s", name)
                self.script("rag/docker/run.bash", "stop", name)

    def start_rag(self, name: str) -> dict:
        if not self.container_running(f"rag_{name}"):
            log.info("starting rag_%s", name)
            subprocess.run(["docker", "rm", "-f", f"rag_{name}"], capture_output=True)  # a stopped container
            self.script("rag/docker/run.bash", "start", name)
        start = time.monotonic()
        while time.monotonic() - start < self.cfg.rag_timeout:
            health = self.rag_health(name)
            if health and health.get("loaded"):
                return health
            if not self.container_running(f"rag_{name}"):
                raise LoadError(f"rag_{name} stopped while starting: bash rag/docker/run.bash logs {name}")
            time.sleep(3)
        raise LoadError(f"rag_{name} not ready after {self.cfg.rag_timeout:.0f} s")

    def ensure_ollama(self):
        if self.ollama_models() is None:
            log.info("starting Ollama")
            self.script("ollama/docker/run.bash", "start")
            for _ in range(30):
                if self.ollama_models() is not None:
                    return
                time.sleep(2)
            raise LoadError(f"Ollama does not answer at {self.cfg.ollama_url}")

    def unload_models(self, keep: str = ""):
        for model in self.ollama_models() or []:
            if model != keep:
                log.info("unloading %s from Ollama", model)
                http_json(f"{self.cfg.ollama_url}/api/generate", {"model": model, "keep_alive": 0}, timeout=60)

    def load_model(self, model: str, options: dict):
        """Loads the model in Ollama with the context of its requests (another num_ctx would load it again)."""
        body = {"model": model, "keep_alive": self.cfg.keep_alive}
        if options.get("num_ctx"):
            body["options"] = {"num_ctx": int(options["num_ctx"])}
        log.info("loading %s in Ollama (%s)", model, body.get("options", {}))
        try:
            http_json(f"{self.cfg.ollama_url}/api/generate", body, timeout=self.cfg.model_timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise LoadError(f"Ollama could not load {model} (bash ollama/docker/run.bash pull {model}): {detail}")
        except OSError as e:
            raise LoadError(f"Ollama could not load {model}: {e}")

    # --- requests

    def check(self, llm: str, rag: str):
        if not llm and not rag:
            raise HTTPError(400, "give llm (<backend>:<model>) and/or rag (<name>)")
        if rag and rag not in self.rag_names():
            raise HTTPError(400, f"unknown RAG {rag}: {self.rag_names()}")
        if llm:
            backend, _, model = llm.partition(":")
            if not model or backend not in ("ollama", "openai", "transformers", "echo"):
                raise HTTPError(400, f"llm must be <backend>:<model>, backend ollama, openai, transformers or echo "
                                     f"(got {llm!r})")

    def load(self, llm: str, rag: str):
        with self._load_lock:
            # the same pair already in memory: nothing to do (without llm, the rag decides the model)
            if self.state == "ready" and rag == self.active["rag"] and llm in ("", self.active["llm"]) \
                    and self._still_loaded():
                return
            with self._state_lock:
                self.state, self.requested, self.error = "loading", {"llm": llm, "rag": rag}, ""
            start = time.monotonic()
            mark("orchestrator", "load", llm=llm, rag=rag)
            try:
                options = {}
                # first free the memory: the other RAGs and the other models
                self.stop_rags(keep=rag)
                if rag or llm.startswith("ollama:"):
                    self.ensure_ollama()
                keep = llm.partition(":")[2] if llm.startswith("ollama:") else ""
                if rag:
                    health = self.start_rag(rag)
                    generator = health.get("generator") or {}
                    pair_llm = f"{generator.get('backend')}:{generator.get('model')}"
                    if llm and llm != pair_llm:
                        raise LoadError(f"RAG {rag} answers with {pair_llm}, not {llm}: give only rag, or change "
                                        f"generator_model in rag_{rag} of servers.yaml")
                    llm, options = pair_llm, generator.get("options") or {}
                    keep = llm.partition(":")[2] if llm.startswith("ollama:") else ""
                self.unload_models(keep=keep)
                if llm.startswith("ollama:"):
                    self.load_model(keep, options)
            except LoadError as e:
                with self._state_lock:
                    self.state, self.error, self.load_s = "failed", str(e), round(time.monotonic() - start, 1)
                log.error("load of %s failed: %s", {"llm": llm, "rag": rag}, e)
                mark("orchestrator", "failed", llm=llm, rag=rag, error=e)
                raise
            with self._state_lock:
                self.state, self.active, self.load_s = "ready", {"llm": llm, "rag": rag}, round(time.monotonic() - start, 1)
            log.info("ready: %s in %.1f s", self.active, self.load_s)
            mark("orchestrator", "ready", llm=llm, rag=rag, load_s=self.load_s)

    def _still_loaded(self) -> bool:
        """The active pair is still in memory (Ollama unloads a model after keep_alive, a container can stop)."""
        llm, rag = self.active["llm"], self.active["rag"]
        if rag and not (self.rag_health(rag) or {}).get("loaded"):
            return False
        return not llm.startswith("ollama:") or llm.partition(":")[2] in (self.ollama_models() or [])

    def unload(self):
        with self._load_lock:
            self.stop_rags()
            if self.ollama_models() is not None:
                self.unload_models()
            with self._state_lock:
                self.state, self.active, self.requested, self.error = "idle", {"llm": "", "rag": ""}, None, ""
            mark("orchestrator", "unload")

    def status(self) -> dict:
        rags = {}
        for name in self.rag_names():
            health = self.rag_health(name) if self.container_running(f"rag_{name}") else None
            rags[name] = {"running": health is not None, "ready": bool(health and health.get("loaded"))}
        try:
            llm_server = http_json(f"{self.cfg.llm_url}/health", timeout=3).get("status") == "ok"
        except (OSError, ValueError):
            llm_server = False
        with self._state_lock:
            return {"state": self.state, "active": dict(self.active), "requested": self.requested, "error": self.error,
                    "load_s": self.load_s, "ollama": self.ollama_models(), "rags": rags, "llm_server": llm_server}


class OrchestratorService(HTTPService):

    def __init__(self, orchestrator: Orchestrator):
        super().__init__("orchestrator", orchestrator.cfg.host, orchestrator.cfg.port)
        self.orchestrator = orchestrator
        self.add_route("GET", "/status", lambda request: self.orchestrator.status())
        self.add_route("POST", "/load", self.load)
        self.add_route("POST", "/unload", self.unload)

    def health(self) -> dict:
        return {"state": self.orchestrator.state, "active": self.orchestrator.active}

    def load(self, request) -> dict:
        data = request.read_json()
        llm, rag = str(data.get("llm") or "").strip(), str(data.get("rag") or "").strip()
        self.orchestrator.check(llm, rag)
        if not data.get("wait", True):
            if self.orchestrator.state == "loading":
                raise HTTPError(409, f"a load is running: {self.orchestrator.requested} (GET /status)")
            threading.Thread(target=self._load_quietly, args=(llm, rag), daemon=True).start()
            time.sleep(0.2)  # the thread takes the state
            return self.orchestrator.status()
        try:
            self.orchestrator.load(llm, rag)
        except LoadError as e:
            raise HTTPError(502, str(e))
        return self.orchestrator.status()

    def _load_quietly(self, llm, rag):
        try:
            self.orchestrator.load(llm, rag)
        except LoadError:
            pass  # in /status: state failed, error

    def unload(self, request) -> dict:
        self.orchestrator.unload()
        return self.orchestrator.status()


def main(argv=None):
    setup_logging()
    cfg = config.load(argv)
    orchestrator = Orchestrator(cfg)
    service = OrchestratorService(orchestrator)
    if cfg.default_llm or cfg.default_rag:  # the default pair, in the background: the server answers meanwhile
        orchestrator.check(cfg.default_llm, cfg.default_rag)
        threading.Thread(target=service._load_quietly, args=(cfg.default_llm, cfg.default_rag), daemon=True).start()
    service.serve_forever()


if __name__ == "__main__":
    main()

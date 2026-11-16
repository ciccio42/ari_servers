"""Piper text to speech server: the audio of the answers of the demo over HTTP, for the behaviour trees of ARI.

The voices are loaded once at the start (and warmed up: the first synthesis on the GPU takes about 1 s). One request
at a time among all the servers of the machine (execution lock, tts.exclusive), during the synthesis.

  GET  /health      {"status": "ok", "service": "tts_server", "voices", "default_voice", "provider", "loaded"}
  POST /synthesize  {"text": "...", "voice": "Italian" (optional: default the first voice of tts.voices)}
                    reply: audio/wav (16 bit mono PCM, sample rate of the voice), with the headers
                    X-Voice, X-Audio-S (length of the audio), X-Queue-S (wait for the lock), X-Synthesis-S, X-Total-S

Errors: 400 bad request (empty or too long text, unknown voice), 500 synthesis failed. The server keeps running.

Run: python -m tts_server [--use-cuda false] [--port ...] (PYTHONPATH: common and tts).
"""
import io
import time
import wave

from ari_servers import execution
from ari_servers.config import workspace_path
from ari_servers.execution import ExclusiveExecution
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import config


def parse_voices(text: str) -> dict:
    """"Italian=models/a.onnx, English=models/b.onnx" -> {"Italian": Path, "English": Path}, in this order."""
    voices = {}
    for item in text.split(","):
        if item.strip():
            name, _, path = item.partition("=")
            if not path.strip():
                raise SystemExit(f"tts.voices: '{item.strip()}' is not <name>=<model.onnx>")
            voices[name.strip()] = workspace_path(path.strip())
    if not voices:
        raise SystemExit("tts.voices is empty")
    return voices


class TTSService(HTTPService):

    def __init__(self, cfg: config.TTSConfig, exclusive: ExclusiveExecution):
        super().__init__("tts_server", cfg.host, cfg.port, exclusive if cfg.exclusive else ExclusiveExecution(None))
        from piper import PiperVoice  # only here: the module of the configuration does not need it

        self.cfg = cfg
        self.voices = {}
        for name, path in parse_voices(cfg.voices).items():
            if not path.is_file():
                raise SystemExit(f"voice {name}: model {path} not found")
            t0 = time.perf_counter()
            voice = PiperVoice.load(str(path), use_cuda=cfg.use_cuda)
            self._synthesize(voice, "riscaldamento")  # warm-up: CUDA initializes at the first run
            self.voices[name] = voice
            self.log.info("voice %s: %s loaded in %.2f s", name, path.name, time.perf_counter() - t0)
        self.default_voice = next(iter(self.voices))
        self.add_route("POST", "/synthesize", self.synthesize)

    def health(self) -> dict:
        return {"voices": list(self.voices), "default_voice": self.default_voice,
                "provider": "cuda" if self.cfg.use_cuda else "cpu", "loaded": True}

    @staticmethod
    def _synthesize(voice, text: str) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()

    def synthesize(self, request):
        data = request.read_json()
        text = " ".join(str(data.get("text") or "").split())
        if not text:
            raise HTTPError(400, "empty text")
        if len(text) > self.cfg.max_chars:
            raise HTTPError(400, f"text longer than {self.cfg.max_chars} characters")
        name = data.get("voice") or self.default_voice
        voice = self.voices.get(name)
        if voice is None:
            raise HTTPError(400, f"unknown voice {name}: {list(self.voices)}")

        with self.execution.hold() as waited:
            t_synthesis = time.perf_counter()
            audio = self._synthesize(voice, text)
            t_end = time.perf_counter()
        with wave.open(io.BytesIO(audio)) as wav_file:
            audio_s = wav_file.getnframes() / wav_file.getframerate()

        timing = {"X-Voice": name, "X-Audio-S": f"{audio_s:.3f}", "X-Queue-S": f"{waited:.4f}",
                  "X-Synthesis-S": f"{t_end - t_synthesis:.4f}", "X-Total-S": f"{t_end - request.t_in:.4f}"}
        request.reply_raw(200, audio, "audio/wav", timing)
        mark("tts_server", "done", client=request.client_address[0], voice=name, chars=len(text), audio_s=audio_s,
             bytes=len(audio), queue_s=timing["X-Queue-S"], synthesis_s=timing["X-Synthesis-S"],
             total_s=timing["X-Total-S"])
        return None  # already replied


def main(argv=None):
    setup_logging()
    TTSService(config.load(argv), execution.from_config()).serve_forever()


if __name__ == "__main__":
    main()

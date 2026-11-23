"""Piper text to speech server: the audio of the answers of the demo over HTTP, for the behaviour trees of ARI.

The voices are loaded once at the start (and warmed up: the first synthesis on the GPU takes about 1 s). One request
at a time among all the servers of the machine (execution lock, tts.exclusive), during the synthesis.

  GET  /health      {"status": "ok", "service": "tts_server", "voices", "default_voice", "provider", "loaded"}
  POST /synthesize  {"text": "...", "voice": "Italian" (optional: default the first voice of tts.voices),
                     "format": "wav" (default) or "opus"}
                    reply: audio/wav (16 bit mono PCM, sample rate of the voice) or, with "format": "opus",
                    audio/ogg (Ogg Opus, about 32 kbit/s, 24 kHz: about 1/10 of the WAV, for slow networks; the
                    encoding costs about 25 ms per second of audio). Headers: X-Voice, X-Audio-Format (wav|opus),
                    X-Audio-S (length of the audio), X-Queue-S (wait for the lock), X-Synthesis-S, X-Encode-S
                    (Opus encoding, 0 for WAV), X-Total-S

Errors: 400 bad request (empty or too long text, unknown voice), 500 synthesis failed. The server keeps running.

Run: python -m tts_server [--use-cuda false] [--port ...] (PYTHONPATH: common and tts).
"""
import io
import time
import wave

import numpy as np
import soundfile
from scipy.signal import resample_poly

from ari_servers import execution
from ari_servers.config import workspace_path
from ari_servers.execution import ExclusiveExecution
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import config

AUDIO_FORMATS = ("wav", "opus")
# Opus takes only these rates: the audio of the voice (22.05 kHz for the medium voices) is resampled to the next one
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)
# compression level of libsndfile for Opus: 0.9 gives about 32 kbit/s on the voice of Piper
OPUS_COMPRESSION_LEVEL = 0.9


def wav_to_opus(wav: bytes) -> bytes:
    """Ogg Opus of a mono 16-bit WAV."""
    with wave.open(io.BytesIO(wav)) as wav_file:
        rate = wav_file.getframerate()
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    target = next((r for r in OPUS_RATES if r >= rate), OPUS_RATES[-1])
    if target != rate:
        g = np.gcd(rate, target)
        samples = np.clip(resample_poly(samples.astype(np.float32), target // g, rate // g), -32768, 32767)
        samples = samples.astype(np.int16)
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, target, format="OGG", subtype="OPUS", compression_level=OPUS_COMPRESSION_LEVEL)
    return buffer.getvalue()


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
                "provider": "cuda" if self.cfg.use_cuda else "cpu", "loaded": True,
                "audio_formats": list(AUDIO_FORMATS)}

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
        fmt = str(data.get("format") or "wav").lower()
        if fmt not in AUDIO_FORMATS:
            raise HTTPError(400, f"unknown format {fmt}: {', '.join(AUDIO_FORMATS)}")

        with self.execution.hold() as waited:
            t_synthesis = time.perf_counter()
            audio = self._synthesize(voice, text)
            t_end = time.perf_counter()
        with wave.open(io.BytesIO(audio)) as wav_file:
            audio_s = wav_file.getnframes() / wav_file.getframerate()
        # outside the execution lock: the encoding runs on the CPU and does not need the GPU
        wav_bytes = len(audio)
        if fmt == "opus":
            audio = wav_to_opus(audio)
        t_encoded = time.perf_counter()

        timing = {"X-Voice": name, "X-Audio-Format": fmt, "X-Audio-S": f"{audio_s:.3f}", "X-Queue-S": f"{waited:.4f}",
                  "X-Synthesis-S": f"{t_end - t_synthesis:.4f}", "X-Encode-S": f"{t_encoded - t_end:.4f}",
                  "X-Total-S": f"{t_encoded - request.t_in:.4f}"}
        request.reply_raw(200, audio, "audio/ogg" if fmt == "opus" else "audio/wav", timing)
        mark("tts_server", "done", client=request.client_address[0], voice=name, chars=len(text), audio_s=audio_s,
             audio_format=fmt, bytes=len(audio), wav_bytes=wav_bytes, queue_s=timing["X-Queue-S"],
             synthesis_s=timing["X-Synthesis-S"], encode_s=timing["X-Encode-S"], total_s=timing["X-Total-S"])
        return None  # already replied


def main(argv=None):
    setup_logging()
    TTSService(config.load(argv), execution.from_config()).serve_forever()


if __name__ == "__main__":
    main()

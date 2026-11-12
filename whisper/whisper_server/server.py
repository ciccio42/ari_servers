"""Whisper over the network: HTTP server that transcribes the audio sent by its clients (no ROS).

The ROS nodes of the demo reach it through whisper_client (ari_control / ari_demo); a script or curl works too.
The model is a parameter (WhisperConfig: servers.yaml, WHISPER_<FIELD>, --<field>). A request transcribes only
when no other server of the machine is running one (ExclusiveExecution): the wait is queue_s in the reply.

  GET  /health      {"status": "ok", "service": "whisper", "model": ..., "model_dir": ..., "language": ..., ...}
  POST /transcribe  body: mono float32 little-endian samples at 16 kHz (Content-Type: application/octet-stream)
                    or a WAV file (Content-Type: audio/wav), mono, 16 kHz, 16-bit PCM
                    optional header X-VAD-Threshold: VAD threshold of this request
                    reply: {"text", "speech", "samples", "audio_s", "decode_s", "queue_s", "vad_s", "asr_s", "total_s"}

Run: python -m whisper_server [--model ...] [--port ...] (PYTHONPATH: common and whisper). Every request writes a
whisper_server.done line in the latency log, with the durations measured on this machine.
"""
import io
import time
import wave

import numpy as np

from ari_servers import execution
from ari_servers.execution import ExclusiveExecution
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import config
from .model import SAMPLING_RATE, WhisperTranscriber

WAV_TYPES = ("audio/wav", "audio/x-wav", "audio/wave")


def decode_wav(body: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(body)) as f:
        if f.getnchannels() != 1 or f.getframerate() != SAMPLING_RATE:
            raise HTTPError(400, f"WAV must be mono at {SAMPLING_RATE} Hz, got {f.getnchannels()} channels at {f.getframerate()} Hz")
        frames, width = f.readframes(f.getnframes()), f.getsampwidth()
    if width != 2:
        raise HTTPError(400, f"WAV sample width {width} not supported: use 16-bit PCM, or raw float32")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def decode_raw(body: bytes) -> np.ndarray:
    if len(body) % 4:
        raise HTTPError(400, "raw audio must be float32: body length not a multiple of 4")
    return np.frombuffer(body, dtype="<f4").copy()


class WhisperService(HTTPService):

    def __init__(self, cfg: config.WhisperConfig, exclusive: ExclusiveExecution):
        super().__init__("whisper_server", cfg.host, cfg.port, exclusive)
        self.cfg = cfg
        self.model = WhisperTranscriber(cfg, logger=self.log)
        self.add_route("POST", "/transcribe", self.transcribe)

    def health(self) -> dict:
        return {"model": self.model.model_id, "model_dir": str(self.model.model_path), "language": self.model.language,
                "vad_threshold": self.model.vad_threshold, "optimization": self.cfg.optimization,
                "provider": self.cfg.provider}

    def transcribe(self, request) -> dict:
        body = request.read_body(self.cfg.max_body_mb * 1024 * 1024)
        content_type = request.headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()
        try:
            audio = decode_wav(body) if content_type in WAV_TYPES else decode_raw(body)
            threshold = request.headers.get("X-VAD-Threshold")
            threshold = float(threshold) if threshold not in (None, "") else None
        except (ValueError, wave.Error, EOFError) as e:
            raise HTTPError(400, str(e) or f"invalid audio ({type(e).__name__})")
        if audio.size == 0:
            raise HTTPError(400, "empty audio")
        t_decoded = time.perf_counter()
        with self.execution.hold() as waited:
            result = self.model.transcribe(audio, vad_threshold=threshold)
        t_end = time.perf_counter()
        result.update(decode_s=round(t_decoded - request.t_in, 4), queue_s=round(waited, 4),
                      total_s=round(t_end - request.t_in, 4))
        mark("whisper_server", "done", client=request.client_address[0], samples=result["samples"],
             speech=result["speech"], words=len(result["text"].split()), decode_s=result["decode_s"],
             queue_s=result["queue_s"], vad_s=result["vad_s"], asr_s=result["asr_s"], total_s=result["total_s"])
        self.log.info("%s: %.2f s audio in %.3f s: %s", request.client_address[0], result["audio_s"],
                      result["total_s"], result["text"])
        return result


def main(argv=None):
    setup_logging()
    cfg = config.load(argv)
    WhisperService(cfg, execution.from_config()).serve_forever()


if __name__ == "__main__":
    main()

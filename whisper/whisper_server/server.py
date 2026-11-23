"""Whisper over the network: HTTP server that transcribes the audio sent by its clients (no ROS).

The ROS nodes of the demo reach it through whisper_client (ari_control / ari_demo); a script or curl works too.
The model is a parameter (WhisperConfig: servers.yaml, WHISPER_<FIELD>, --<field>). A request transcribes only
when no other server of the machine is running one (ExclusiveExecution): the wait is queue_s in the reply.

  GET  /health      {"status": "ok", "service": "whisper", "model": ..., "model_dir": ..., "language": ..., ...}
  POST /transcribe  body: mono audio at 16 kHz, the format given by the header X-Audio-Format:
                      f32le (default, also without the header): raw float32 little-endian samples
                      s16le: raw int16 little-endian samples (half the bytes of f32le, same text)
                      ogg:   an Ogg Opus file (Content-Type: audio/ogg also selects it), for slow networks
                    or a WAV file (Content-Type: audio/wav), mono, 16 kHz, 16-bit PCM
                    optional header X-VAD-Threshold: VAD threshold of this request
                    reply: {"text", "speech", "samples", "audio_s", "audio_format", "body_bytes", "receive_s",
                            "decode_s", "queue_s", "vad_s", "asr_s", "total_s"}

Times: receive_s is the read of the body (the upload of the audio, network); total_s starts after it, so a slow
upload is not counted as time of the server (http_s - total_s on the client is all the communication).

Run: python -m whisper_server [--model ...] [--port ...] (PYTHONPATH: common and whisper). Every request writes a
whisper_server.done line in the latency log, with the durations measured on this machine.
"""
import io
import time
import wave

import numpy as np
import soundfile
from scipy.signal import resample_poly

from ari_servers import execution
from ari_servers.execution import ExclusiveExecution
from ari_servers.http import HTTPError, HTTPService, setup_logging
from ari_servers.latency_log import mark

from . import config
from .model import SAMPLING_RATE, WhisperTranscriber

WAV_TYPES = ("audio/wav", "audio/x-wav", "audio/wave")
OGG_TYPES = ("audio/ogg", "audio/opus", "application/ogg")
# values of X-Audio-Format: raw formats and the decoder of each one
AUDIO_FORMATS = ("f32le", "s16le", "ogg", "wav")


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


def decode_s16le(body: bytes) -> np.ndarray:
    if len(body) % 2:
        raise HTTPError(400, "raw audio s16le: body length not a multiple of 2")
    return np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0


def decode_ogg(body: bytes) -> np.ndarray:
    """Ogg Opus (or Vorbis) with libsndfile; mixed to mono and resampled to 16 kHz if the file has another rate."""
    try:
        audio, rate = soundfile.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except (RuntimeError, soundfile.LibsndfileError) as e:
        raise HTTPError(400, f"invalid Ogg audio: {e}")
    audio = audio.mean(axis=1)
    if rate != SAMPLING_RATE:
        g = np.gcd(int(rate), SAMPLING_RATE)
        audio = resample_poly(audio, SAMPLING_RATE // g, int(rate) // g).astype(np.float32)
    return audio


def audio_format(headers) -> str:
    """Format of the body: X-Audio-Format, else the Content-Type (WAV, Ogg), else f32le (the old clients)."""
    value = (headers.get("X-Audio-Format") or "").strip().lower()
    if value:
        value = {"float32": "f32le", "int16": "s16le", "opus": "ogg"}.get(value, value)
        if value not in AUDIO_FORMATS:
            raise HTTPError(400, f"X-Audio-Format {value} not supported: {', '.join(AUDIO_FORMATS)}")
        return value
    content_type = headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()
    return "wav" if content_type in WAV_TYPES else "ogg" if content_type in OGG_TYPES else "f32le"


DECODERS = {"f32le": decode_raw, "s16le": decode_s16le, "ogg": decode_ogg, "wav": decode_wav}


class WhisperService(HTTPService):

    def __init__(self, cfg: config.WhisperConfig, exclusive: ExclusiveExecution):
        super().__init__("whisper_server", cfg.host, cfg.port, exclusive)
        self.cfg = cfg
        self.model = WhisperTranscriber(cfg, logger=self.log)
        self.add_route("POST", "/transcribe", self.transcribe)

    def health(self) -> dict:
        return {"model": self.model.model_id, "model_dir": str(self.model.model_path), "language": self.model.language,
                "vad_threshold": self.model.vad_threshold, "optimization": self.cfg.optimization,
                "provider": self.cfg.provider, "audio_formats": list(AUDIO_FORMATS)}

    def transcribe(self, request) -> dict:
        body = request.read_body(self.cfg.max_body_mb * 1024 * 1024)
        # the timer of the server starts here: the upload of the body is network time, not time of the server
        t_received = time.perf_counter()
        fmt = audio_format(request.headers)
        try:
            audio = DECODERS[fmt](body)
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
        result.update(audio_format=fmt, body_bytes=len(body), receive_s=round(t_received - request.t_in, 4),
                      decode_s=round(t_decoded - t_received, 4), queue_s=round(waited, 4),
                      total_s=round(t_end - t_received, 4))
        mark("whisper_server", "done", client=request.client_address[0], samples=result["samples"],
             speech=result["speech"], words=len(result["text"].split()), audio_format=fmt, body_bytes=len(body),
             receive_s=result["receive_s"], decode_s=result["decode_s"],
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

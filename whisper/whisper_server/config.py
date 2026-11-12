"""Configuration of the Whisper server: section `whisper` of config/servers.yaml, WHISPER_<FIELD>, --<field>."""
from dataclasses import dataclass
from typing import Optional

from ari_servers import config


@dataclass
class WhisperConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # any Whisper model of Hugging Face: openai/whisper-large-v3-turbo, openai/whisper-large-v3, openai/whisper-small...
    model: str = "openai/whisper-large-v3-turbo"
    # exported ONNX model; default: models/whisper/<model name>, plus -<optimization> if not O4.
    # Missing: the model is downloaded, exported and saved there at the first start
    model_dir: Optional[str] = None
    # ONNX Runtime optimization of the export: O1, O2, O3, O4 (O4: fp16, GPU only) or none
    optimization: str = "O4"
    provider: str = "CUDAExecutionProvider"
    language: str = "Italian"  # Italian, English or a code (it, en)
    vad_threshold: float = 0.7  # Silero VAD, overridden per request by the header X-VAD-Threshold
    vad_model: str = "models/silero/silero_vad.onnx"
    warmup: int = 3  # transcriptions of noise at the start, so the first request is not slow
    max_body_mb: int = 64  # about 17 minutes of float32 audio at 16 kHz


ALIASES = {"language": ("DEMO_LANGUAGE",), "vad_threshold": ("SILERO_VAD_THRESHOLD",)}  # names of ari_demo


def load(argv=None) -> WhisperConfig:
    return config.load(WhisperConfig, "whisper", "WHISPER", argv, ALIASES, description="Whisper HTTP server")

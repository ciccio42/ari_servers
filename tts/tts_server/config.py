"""Configuration of the Piper TTS server: section `tts` of config/servers.yaml, TTS_<FIELD>, --<field>."""
from dataclasses import dataclass

from ari_servers import config


@dataclass
class TTSConfig:
    host: str = "0.0.0.0"
    port: int = 8020
    # <voice name>=<Piper model (.onnx, with its .onnx.json next to it)>, comma separated. The requests choose one by
    # name ("voice": "Italian"); the first one is the default
    voices: str = ("Italian=models/piper/it_IT-paola-medium.onnx, "
                   "English=models/piper/en_GB-cori-medium.onnx")
    # true: onnxruntime on the GPU (CUDA). About 0.1 s for any length on the Spark; CPU: 0.08 s for a short phrase,
    # 0.3 s for an answer of 250 characters
    use_cuda: bool = True
    # true: the synthesis takes the execution lock shared with Whisper and the LLM server (no concurrent executions)
    exclusive: bool = True
    max_chars: int = 2000  # longer texts are refused (400)


def load(argv=None) -> TTSConfig:
    return config.load(TTSConfig, "tts", "TTS", argv, description="Piper text to speech HTTP server")

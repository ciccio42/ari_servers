"""Whisper (ONNX Runtime) with Silero VAD: the model of whisper_server, chosen by WhisperConfig.

At the first start the exported model is not in model_dir: it is downloaded from Hugging Face, exported to ONNX,
optimized (WhisperConfig.optimization) and saved there. The next starts load it from the disk.
Same logic of ari_demo (ros2_ws/src/asr/asr/whisper_model.py), with the model as a parameter.
"""
import logging
import time
from pathlib import Path

import numpy as np
import torch
from optimum.onnxruntime import AutoOptimizationConfig, ORTModelForSpeechSeq2Seq, ORTOptimizer
from silero_vad.utils_vad import OnnxWrapper, get_speech_timestamps
from transformers import AutoProcessor

from ari_servers.config import WORKSPACE, workspace_path

from .config import WhisperConfig

SAMPLING_RATE = 16000
LANGUAGES = {"italian": "it", "english": "en"}
OPTIMIZATIONS = ("O1", "O2", "O3", "O4", "none")


def language_code(language: str) -> str:
    language = language.lower()
    if language in LANGUAGES.values():
        return language
    if language not in LANGUAGES:
        raise ValueError(f"{language} not available. Languages: {list(LANGUAGES)} or {list(LANGUAGES.values())}")
    return LANGUAGES[language]


def default_model_dir(cfg: WhisperConfig) -> Path:
    name = cfg.model.rstrip("/").split("/")[-1]
    # O4 without suffix: the folder of the model exported by ari_demo (whisper-large-v3-turbo)
    return WORKSPACE / "models" / "whisper" / (name if cfg.optimization == "O4" else f"{name}-{cfg.optimization.lower()}")


class WhisperTranscriber:

    def __init__(self, cfg: WhisperConfig, logger=None):
        if cfg.optimization not in OPTIMIZATIONS:
            raise ValueError(f"optimization {cfg.optimization} not valid: {OPTIMIZATIONS}")
        self.cfg = cfg
        self.log = logger or logging.getLogger("whisper")
        self.model_id = cfg.model
        self.model_path = workspace_path(cfg.model_dir) if cfg.model_dir else default_model_dir(cfg)
        self.language = language_code(cfg.language)
        self.vad_threshold = cfg.vad_threshold
        self.log.info("Loading Whisper %s from %s (%s, %s)", self.model_id, self.model_path, cfg.optimization, cfg.provider)
        self._load_processor()
        self._load_model()
        self._load_vad_model(workspace_path(cfg.vad_model))
        for _ in range(cfg.warmup):
            self.predict(torch.rand(SAMPLING_RATE * 10).numpy())
        self.log.info("ASR model loaded")

    def _load_processor(self):
        try:
            AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            processor = AutoProcessor.from_pretrained(self.model_id)
            self.model_path.mkdir(exist_ok=True, parents=True)
            processor.save_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.set_prefix_tokens(language=self.language, task="transcribe")

    def _load_model(self):
        try:
            self.model = ORTModelForSpeechSeq2Seq.from_pretrained(self.model_path, provider=self.cfg.provider)
        except ValueError:  # model not exported yet
            self.log.info("Exporting %s to ONNX in %s (optimization %s)", self.model_id, self.model_path, self.cfg.optimization)
            model = ORTModelForSpeechSeq2Seq.from_pretrained(self.model_id, export=True)
            if self.cfg.optimization == "none":
                model.save_pretrained(self.model_path)
            else:
                optimization = getattr(AutoOptimizationConfig, self.cfg.optimization)()
                ORTOptimizer.from_pretrained(model).optimize(save_dir=self.model_path, optimization_config=optimization)
            self.model = ORTModelForSpeechSeq2Seq.from_pretrained(self.model_path, provider=self.cfg.provider)

    def _load_vad_model(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {path} (it is not downloaded)")
        self.vad_model = OnnxWrapper(path.as_posix())
        get_speech_timestamps(torch.zeros(SAMPLING_RATE * 5, dtype=torch.float32), self.vad_model)
        self.log.info("Silero VAD loaded")

    def predict(self, audio: np.ndarray) -> str:
        features = self.processor(audio, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features
        outputs = self.model.generate(
            features.to(self.model.device),
            task="transcribe",
            language=self.language,
            is_multilingual=True,
            return_dict_in_generate=False,
        )
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        self.log.info(f"ASR Output: {text}")
        return text

    def transcribe(self, audio: np.ndarray, vad_threshold=None) -> dict:
        """audio: mono float32 samples at 16 kHz. Returns text, speech and the durations of VAD and inference."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        threshold = self.vad_threshold if vad_threshold is None else vad_threshold
        t_vad = time.perf_counter()
        speech = bool(get_speech_timestamps(audio, self.vad_model, threshold=threshold))
        t_asr = time.perf_counter()
        if speech:
            text = self.predict(audio)
        else:
            self.log.info("Silero VAD: no speech")
            text = ""
        t_end = time.perf_counter()
        return {"text": str(text), "speech": speech, "samples": int(audio.size),
                "audio_s": round(audio.size / SAMPLING_RATE, 4),
                "vad_s": round(t_asr - t_vad, 4), "asr_s": round(t_end - t_asr, 4)}

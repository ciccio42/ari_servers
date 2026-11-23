# TTS server (Piper)

The audio of the answers of the demo over HTTP (`<TTS_PORT>`, default 8020): the behaviour trees of ARI send the
text and play the WAV, instead of synthesizing it with Piper on the CPU of the robot. Same voices as ARI
(`it_IT-paola-medium`, `en_GB-cori-medium`), onnxruntime on the GPU.

```
 ARI: TtsNode (BT) ──HTTP POST /synthesize──► tts_server :8020 (Piper, GPU, execution lock)
         │  ◄──────────── audio/wav ─────────────┘
         └─► Tts goal (text + WAV) ─► AriTextToSpeechActionServer ─► AriTTSSpeak (ari_tts_action_server): plays it
```

| Path | Content |
|---|---|
| `tts_server/server.py`, `config.py` | HTTP server, `TTSConfig` |
| `docker/` | image `ari_tts_server` (base `ari_whisper_server:spark`, plus `piper-tts` without its dependencies, so the onnxruntime with CUDA stays), `entrypoint.bash`, `run.bash` |
| `../models/piper/` | the voices (`.onnx` + `.onnx.json`), copied from `ari_control/ros2_ws/src/ari_tts_action_server/models`; the `.onnx` files are not versioned |

## API
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service": "tts_server", "voices", "default_voice", "provider", "loaded"}` |
| `POST /synthesize` | `{"text": "...", "voice": "Italian", "format": "wav"}` (`voice` optional: default the first one of `tts.voices`; `format` optional: `wav` (default) or `opus`) |
| reply | `audio/wav` (16 bit mono PCM, 22050 Hz) or, with `"format": "opus"`, `audio/ogg` (Ogg Opus 24 kHz, about 32 kbit/s, about 1/10 of the WAV: for slow networks, `AUDIO_CODEC=opus` on ARI; the encoding takes about 25 ms per second of audio, on the CPU, outside the execution lock). Headers `X-Voice`, `X-Audio-Format`, `X-Audio-S` (audio length), `X-Queue-S` (wait for the execution lock), `X-Synthesis-S`, `X-Encode-S` (Opus encoding, 0 for WAV), `X-Total-S` |
| errors | `400` empty or too long text, unknown voice; `500` synthesis failed |

```bash
curl -X POST -H 'Content-Type: application/json' -o answer.wav localhost:8020/synthesize \
  -d '{"text": "Che bella domanda. Lasciami pensare", "voice": "Italian"}'
```

## Configuration
Section `tts` of [config/servers.yaml](../config/servers.yaml), environment `TTS_<FIELD>`, option `--<field>`:

| Field | Default | |
|---|---|---|
| `port` | 8020 | |
| `voices` | `Italian=models/piper/it_IT-paola-medium.onnx, English=models/piper/en_GB-cori-medium.onnx` | `<name>=<model>`: the requests choose the voice by name (ARI sends `DEMO_LANGUAGE`) |
| `use_cuda` | `true` | GPU; `false`: CPU |
| `exclusive` | `true` | the synthesis takes the execution lock shared with Whisper and the LLM server |
| `max_chars` | 2000 | longer texts: 400 |

## Run with Docker (Spark)
From `ari_servers` (or with `bash run_servers.bash start tts`):
```bash
bash tts/docker/run.bash build       # image ari_tts_server, once (needs ari_whisper_server:spark)
bash tts/docker/run.bash start       # container tts_server, restarts with the machine; ready in about 5 s
bash tts/docker/run.bash status      # /health
bash tts/docker/run.bash test "Ciao, sono ARI"    # audio in /tmp/tts_test.wav, timing headers printed
bash tts/docker/run.bash logs
bash tts/docker/run.bash stop
```
After a change of the code or of the configuration: `stop`, then `start`. Every request writes a `tts_server.done` line
in `latency_logs/latency_<date>_<host>.csv`.

## Measures (2026-11-13)
Piper `it_IT-paola-medium`, median of 7 runs, after the warm-up.

Synthesis on the Spark, CPU vs GPU:

| Text | Audio | CPU | GPU (CUDA) |
|---|---:|---:|---:|
| short (35 characters, a waiting phrase) | 2.0 s | 76 ms | 93 ms |
| medium (88 characters) | 4.4 s | 132 ms | 96 ms |
| long (242 characters, an answer of the LLM) | 12.8 s | 299 ms | 106 ms |

On the GPU the time hardly grows with the length of the text; the first synthesis takes about 1 s (CUDA
initialization), done at the start of the server.

From ARI (Wi-Fi), the request with a persistent connection (what `TtsNode` does) against Piper on the CPU of ARI:

| Text | WAV | Piper on ARI | Spark: synthesis | Spark: whole request from ARI |
|---|---:|---:|---:|---:|
| short | 86 KB | 199 ms | 87 ms | 95 ms |
| medium | 179 KB | 436 ms | 89 ms | 104 ms |
| long | 553 KB | 1227 ms | 99 ms | 126 ms |

A new connection for every request costs about 20 ms more. The fixed phrases already cached on ARI are not asked here.

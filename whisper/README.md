# Whisper server

HTTP server with Whisper (ONNX Runtime, CUDA) and Silero VAD, with the **model as a parameter** (default
`openai/whisper-large-v3-turbo`, O4 fp16) and the execution lock shared with the other servers.
```
 robot                                  machine with the GPU
 client ──HTTP POST /transcribe──► whisper_server :8000
                                    (this folder)
```

| Path | Content |
|---|---|
| `whisper_server/` | Python package: `config.py` (`WhisperConfig`), `model.py` (Whisper + VAD), `server.py` (HTTP); `python -m whisper_server` (PYTHONPATH: `common` and `whisper`) |
| `docker/` | images `ari_whisper_server:<spark\|orin\|jp5>`, `entrypoint.bash`, `run.bash` |
| `conda/` | machines without Docker (cluster): `cluster/environment.yml`, `run.bash` |
| `tools/` | `bench_http.py` (protocol of the report over HTTP), `summarize_whisper.py` (statistics) |

## API
| | |
|---|---|
| `GET /health` | `{"status": "ok", "service", "model", "model_dir", "language", "vad_threshold", "optimization", "provider"}` |
| `POST /transcribe` | body: mono float32 little-endian samples at 16 kHz (`Content-Type: application/octet-stream`), or a WAV mono 16 kHz 16-bit (`Content-Type: audio/wav`) |
| header `X-VAD-Threshold` | optional, VAD threshold of this request |
| reply | `{"text", "speech", "samples", "audio_s", "decode_s", "queue_s", "vad_s", "asr_s", "total_s"}`: seconds measured on the server, `decode_s` includes the reception of the body |
| errors | `400` bad audio, `404` unknown path, `500` transcription failed: the server keeps running |

```bash
curl http://<server>:8000/health
curl -X POST -H 'Content-Type: audio/wav' --data-binary @audio_files/speech_20261105_111545_246613.wav \
  http://<server>:8000/transcribe
```

One request at a time, also with the other servers of the machine (execution lock, see the main README): the others
wait (`queue_s`). `TCP_NODELAY` and keep-alive: a client that keeps the connection open saves the connection setup.
**No authentication**: lab network only.

## Model
Options of the section `whisper` of `config/servers.yaml`, or `WHISPER_<FIELD>`, or `--<field>`:

| Field | Default | |
|---|---|---|
| `model` | `openai/whisper-large-v3-turbo` | any Whisper of Hugging Face (`openai/whisper-large-v3`, `openai/whisper-small`...) |
| `model_dir` | `models/whisper/<model name>`, plus `-<optimization>` if not O4 | exported ONNX model |
| `optimization` | `O4` | ONNX Runtime optimization of the export: `O1`..`O4` (O4: fp16, GPU only) or `none` |
| `provider` | `CUDAExecutionProvider` | `CPUExecutionProvider` for tests without GPU (with `optimization` not O4) |
| `language` | `Italian` | `Italian`, `English`, `it`, `en` (also `DEMO_LANGUAGE`) |
| `vad_threshold` | 0.7 | Silero VAD (also `SILERO_VAD_THRESHOLD`); `X-VAD-Threshold` overrides it per request |
| `vad_model` | `models/silero/silero_vad.onnx` | not downloaded |
| `warmup` | 3 | transcriptions at the start, so the first request is not slow |
| `host`, `port` | `0.0.0.0`, 8000 | |

When `model_dir` has no exported model, the server downloads `model`, exports it to ONNX, optimizes it and saves it
there (large-v3-turbo: 1.6 GB, about 4–6 minutes). A model already exported elsewhere can be used instead:
```bash
export WHISPER_MODEL_DIR=<folder of the exported model>   # run.bash mounts it read-only
```
or copied into `models/whisper/`. GPU memory with large-v3-turbo: about 4 GB.

## Run with Docker (Spark, Orin, internal Jetson of ARI)
From `ari_servers`:
```bash
bash whisper/docker/run.bash spark build      # image ari_whisper_server:spark
bash whisper/docker/run.bash spark start      # container whisper_server, in the background, restarts with the machine
bash whisper/docker/run.bash spark logs       # ready at "Whisper server listening on http://0.0.0.0:8000"
bash whisper/docker/run.bash spark test       # /health and 3 transcriptions of a WAV
bash whisper/docker/run.bash spark stop
```
`orin` and `jp5` instead of `spark` on the Jetsons. `server` runs it in the foreground, `shell` opens a shell in the image.

| Board | Base image | Notes |
|---|---|---|
| `spark` | `ari_onnxruntime:spark-r39.1.arm64-sbsa-cu130-24.04` | ONNX Runtime 1.23.2 built for the GB10 with jetson-containers: `build` builds it if missing, with `docker/spark/jetson-containers-cu130.patch` applied to jetson-containers (CUDA 13.0, at most 6 build jobs: with the defaults the build used all the 121 GB of RAM) |
| `orin` | `onnxruntime:r36.4.tegra-aarch64-cu126-22.04` | JetPack 6; torch of jetson-ai-lab, with OpenBLAS and cuDSS added |
| `jp5` | `pepper_demo:r35` | JetPack 5, Python 3.8: the demo image already has torch, ONNX Runtime 1.16.1 and transformers |

The container is named `whisper_server`: if another container with that name is running, stop it first
(`docker stop whisper_server`, then remove it by ID), only one server can listen on the port.

## Run with conda (cluster)
```bash
bash whisper/conda/run.bash cluster create    # environment ari_whisper: whisper/conda/cluster/environment.yml
bash whisper/conda/run.bash cluster start     # in the background, log in latency_logs/whisper_server_<host>.log
bash whisper/conda/run.bash cluster logs
bash whisper/conda/run.bash cluster test
bash whisper/conda/run.bash cluster stop
```
Miniforge first, without root: `bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3`. On the cluster only the
SSH port of the container is reachable from the lab network: clients need an SSH tunnel
(`ssh -p 2678 -N -L 8011:localhost:8000 <user>@172.16.174.236`).

## Environment
Besides `WHISPER_<FIELD>` (table above): `run.bash` and `conda/run.bash` pass them only if set, so they do not hide
`servers.yaml`.

| Variable | Default | |
|---|---|---|
| `WORKSPACE` | `/workspace` (Docker), `ari_servers` (conda) | `config/`, `models/`, `latency_logs/`, `run/` |
| `ARI_SERVERS_CONFIG` | `config/servers.yaml` | |
| `ARI_EXECUTION_LOCK` | `run/execution.lock` | empty: lock disabled |
| `LATENCY_LOG` | True | `False` disables the latency log |

Every request writes a `whisper_server.done` line in `latency_logs/latency_<date>_<host>.csv`.

## Benchmark
Protocol of the report: the 5 WAV alternated, 3 warmup rounds, 20 measured (100 calls).
```bash
python3 whisper/tools/bench_http.py --url http://<server>:8000 --audio-dir audio_files --runs 20 --warmup 3 \
  --output latency_logs/whisper_runs_<client>_to-<server>.csv
python3 whisper/tools/summarize_whisper.py audio_files latency_logs/whisper_runs_<client>_to-<server>.csv
```
`bench_http.py` needs numpy and soundfile; it saves the times of every reply of the server, so the summary needs no
latency log and no clock sync between client and server.

Accuracy: `wer.py` (standard library only) gives WER and CER of the transcripts against
`audio_files/references.csv`, with the normalization of Whisper (lowercase, no punctuation):
```bash
python3 whisper/tools/wer.py latency_logs/whisper_runs_*.csv --per-file --csv latency_logs/wer.csv
```

Checked on the Spark on 2026-11-12, with the libraries of `ari_whisper:spark` and the entrypoint of this folder,
large-v3-turbo O4 model through `WHISPER_MODEL_DIR`: inference 0.184 ± 0.026 s, VAD 0.017 ± 0.003 s (15 calls);
with the execution lock held 2 s by another process, a request waited `queue_s` 1.97 s and then ran. The images `ari_whisper_server:*` are not built yet.

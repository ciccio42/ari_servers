#!/bin/bash
# Whisper HTTP server in a conda environment, without Docker and without ROS (machines where Docker is not
# available, such as the cluster). Same server of the images: whisper/whisper_server, see whisper/README.md.
#
# Usage: bash run.bash <cluster> <create|server|start|stop|status|logs|test> [args]
#   create   creates (or updates) the conda environment of the machine: whisper/conda/<machine>/environment.yml
#   server   runs the server in the foreground: [whisper_server args]
#   start    runs the server in the background (log: latency_logs/whisper_server_<host>.log)
#   stop     stops the server started by `start`
#   status   pid of the server and GET /health
#   logs     follows the log of the server
#   test     GET /health and POST /transcribe of a WAV file (default: the first one of audio_files)
#   bench    protocol of the report against the server, over HTTP: [bench_http.py args] (default 20 runs, 3 warmup)
#
# ari_servers is the workspace (models/, audio_files/, latency_logs/), the server reads config/servers.yaml.
# Environment (only if set, they override servers.yaml): CONDA_ENV (default ari_whisper),
# DEMO_LANGUAGE (default Italian), SILERO_VAD_THRESHOLD (default 0.7), WHISPER_HOST (default 0.0.0.0),
# WHISPER_PORT (default 8000), WHISPER_SERVER_URL (for `test` and `status`, default http://localhost:$WHISPER_PORT)

MACHINE=$1
COMMAND=$2
shift 2 || { sed -n 5,13p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers: models/, audio_files/, latency_logs/
ENV_FILE=$SCRIPT_DIR/$MACHINE/environment.yml
[ -f "$ENV_FILE" ] || { echo "Unknown machine $MACHINE: $ENV_FILE not found" >&2; sed -n 5,13p "$0"; exit 1; }

CONDA_ENV=${CONDA_ENV:-ari_whisper}
CONDA=$(command -v conda || command -v "$HOME/miniforge3/bin/conda" || command -v "$HOME/miniconda3/bin/conda")
[ -n "$CONDA" ] || { echo "conda not found: install Miniforge (see whisper/README.md)" >&2; exit 1; }

export WORKSPACE=$REPO_DIR
export PYTHONPATH=$REPO_DIR/common:$REPO_DIR/whisper${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-$REPO_DIR/models/whisper/.hf_cache}
export PYTHONUNBUFFERED=1
# the server reads config/servers.yaml: WHISPER_<FIELD>, DEMO_LANGUAGE, SILERO_VAD_THRESHOLD override it only if set
URL=${WHISPER_SERVER_URL:-http://localhost:${WHISPER_PORT:-8000}}
LOG_DIR=$REPO_DIR/latency_logs
PID_FILE=$LOG_DIR/whisper_server_$(hostname).pid
LOG_FILE=$LOG_DIR/whisper_server_$(hostname).log

env_python() {
    local prefix
    prefix=$("$CONDA" env list | awk -v env="$CONDA_ENV" '$1 == env {print $NF}')
    [ -x "$prefix/bin/python" ] || { echo "conda environment $CONDA_ENV not found: run '$0 $MACHINE create'" >&2; exit 1; }
    PY=$prefix/bin/python
    # ONNX Runtime loads CUDA and cuDNN from the nvidia-*-cu12 wheels installed with torch
    local libs
    libs=$("$PY" -c 'import glob, os, nvidia; print(":".join(sorted(glob.glob(os.path.join(nvidia.__path__[0], "*", "lib")))))')
    export LD_LIBRARY_PATH=$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
}

running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2> /dev/null
}

health() {
    "$PY" -c 'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1] + "/health", timeout=5).read().decode())' "$URL"
}

case "$COMMAND" in
    create)
        if "$CONDA" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
            "$CONDA" env update -n "$CONDA_ENV" -f "$ENV_FILE" --prune || exit 1
        else
            "$CONDA" env create -n "$CONDA_ENV" -f "$ENV_FILE" || exit 1
        fi
        env_python
        # silero-vad without its dependencies, as in the images: it would bring the CPU onnxruntime and torchaudio
        "$PY" -m pip install --no-deps silero-vad || exit 1
        # a dependency may have brought the CPU onnxruntime: it hides the CUDA provider of onnxruntime-gpu
        if "$PY" -m pip show onnxruntime > /dev/null 2>&1; then
            "$PY" -m pip uninstall -y onnxruntime
            "$PY" -m pip install --force-reinstall --no-deps "$(grep -o 'onnxruntime-gpu==[0-9.]*' "$ENV_FILE")" || exit 1
        fi
        "$PY" -c 'import torch, onnxruntime as ort
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("onnxruntime", ort.__version__, ort.get_available_providers())'
        ;;
    server)
        env_python
        cd "$REPO_DIR" && exec "$PY" -m whisper_server "$@"
        ;;
    start)
        env_python
        running && { echo "whisper_server already running (pid $(cat "$PID_FILE"))"; exit 0; }
        mkdir -p "$LOG_DIR"
        # cd apart: with `cd && setsid ... &` $! is the pid of a subshell, not of the server
        cd "$REPO_DIR" || exit 1
        setsid nohup "$PY" -m whisper_server "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
        echo $! > "$PID_FILE"
        echo "whisper_server started (pid $!), log: $LOG_FILE"
        echo "Ready at \"Whisper server listening on ...\": $0 $MACHINE logs"
        ;;
    stop)
        running || { echo "whisper_server not running"; rm -f "$PID_FILE"; exit 0; }
        kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE" && echo "whisper_server stopped"
        ;;
    status)
        env_python
        if running; then echo "whisper_server running (pid $(cat "$PID_FILE"))"; else echo "whisper_server not started by $0"; fi
        health || exit 1
        ;;
    logs)
        exec tail -n 50 -f "$LOG_FILE"
        ;;
    test)
        env_python
        WAV=${1:-$(ls "$REPO_DIR"/audio_files/*.wav | head -n 1)}
        health || exit 1
        "$PY" - "$URL" "$WAV" << 'EOF'
import json, sys, time, urllib.request
url, wav = sys.argv[1], sys.argv[2]
body = open(wav, "rb").read()
for i in range(3):
    request = urllib.request.Request(url + "/transcribe", data=body, headers={"Content-Type": "audio/wav"})
    t = time.perf_counter()
    reply = json.loads(urllib.request.urlopen(request, timeout=120).read())
    print(f"run {i}: http {time.perf_counter() - t:.3f} s, server {reply['total_s']:.3f} s, asr {reply['asr_s']:.3f} s,"
          f" audio {reply['audio_s']:.2f} s: {reply['text']!r}")
EOF
        ;;
    bench)
        env_python
        health > /dev/null || { echo "whisper_server not reachable at $URL" >&2; exit 1; }
        exec "$PY" "$REPO_DIR/whisper/tools/bench_http.py" --url "$URL" "$@"
        ;;
    *)
        sed -n 5,13p "$0"; exit 1
        ;;
esac

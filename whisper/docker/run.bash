#!/bin/bash
# Builds and runs the Whisper HTTP server in Docker (no ROS). ari_servers is mounted as /workspace.
#
# Usage: bash run.bash <spark|orin|jp5> <build|server|start|stop|logs|status|test|shell> [args]
#   build   builds the image ari_whisper_server:<board> (on the Spark also its base, with jetson-containers)
#   server  runs the server in the foreground: [whisper_server args]
#   start   runs the server in the background (container whisper_server, it restarts with the machine)
#   stop    stops and removes the container whisper_server
#   logs    follows the log of the server
#   status  container state and GET /health
#   test    GET /health and 3 POST /transcribe of a WAV file (default: the first one of audio_files)
#   shell   shell inside the image
#
# The server reads config/servers.yaml (section whisper). Passed to the container only if set here, so they override
# the file: WHISPER_<FIELD> (WHISPER_MODEL, WHISPER_PORT, WHISPER_OPTIMIZATION, ...), DEMO_LANGUAGE,
# SILERO_VAD_THRESHOLD, ARI_EXECUTION_LOCK. WHISPER_MODEL_DIR: an exported model on this machine (e.g. the one of
# ari_demo), mounted read-only. JETSON_CONTAINERS: clone of jetson-containers (default ~/jetson-containers).
# status and test use WHISPER_PORT, default 8000: set it if servers.yaml has another port.

BOARD=$1
COMMAND=$2
shift 2 || { sed -n 4,13p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=ari_whisper_server:$BOARD
CONTAINER=whisper_server
PORT=${WHISPER_PORT:-8000}
URL=http://localhost:$PORT

case "$BOARD" in
    spark)
        GPU_ARGS=(--gpus=all)
        BASE_IMAGE=ari_onnxruntime:spark-r39.1.arm64-sbsa-cu130-24.04
        ;;
    orin)
        GPU_ARGS=(--runtime=nvidia --gpus=all)
        BASE_IMAGE=onnxruntime:r36.4.tegra-aarch64-cu126-22.04
        ;;
    jp5)
        # internal Jetson of ARI (JetPack 5.1.1): the base is the demo image, already on the Jetson
        GPU_ARGS=(--runtime=nvidia --gpus=all)
        BASE_IMAGE=pepper_demo:r35
        ;;
    *)
        sed -n 4,13p "$0"; exit 1
        ;;
esac

RUN_ARGS=("${GPU_ARGS[@]}" --network=host -e HF_HOME=/workspace/models/whisper/.hf_cache -v "$ROOT_DIR":/workspace)
# only the variables set here: otherwise defaults would hide config/servers.yaml
for variable in $(compgen -e | grep -E '^(WHISPER_|DEMO_LANGUAGE$|SILERO_VAD_THRESHOLD$|ARI_EXECUTION_LOCK$)'); do
    [ "$variable" = WHISPER_MODEL_DIR ] || RUN_ARGS+=(-e "$variable=${!variable}")
done
if [ -n "$WHISPER_MODEL_DIR" ]; then
    RUN_ARGS+=(-v "$(cd "$WHISPER_MODEL_DIR" && pwd)":/workspace/models/external/whisper:ro
               -e WHISPER_MODEL_DIR=/workspace/models/external/whisper)
fi

health() {
    curl -sf -m 5 "$URL/health" && echo
}

case "$COMMAND" in
    build)
        if ! docker image inspect "$BASE_IMAGE" > /dev/null 2>&1; then
            [ "$BOARD" = spark ] || { echo "Base image $BASE_IMAGE not found: build it with jetson-containers" >&2; exit 1; }
            # CUDA 13.0 as the driver of the Spark, with spark/jetson-containers-cu130.patch applied
            (cd "${JETSON_CONTAINERS:-$HOME/jetson-containers}" && CUDA_VERSION=13.0 ./build.sh --skip-tests=all \
                --name=ari_onnxruntime:spark onnxruntime:1.23.2) || exit 1
        fi
        exec docker build -f "$SCRIPT_DIR/$BOARD/Dockerfile" -t "$IMAGE" "$SCRIPT_DIR"
        ;;
    server)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server "$@"
        ;;
    start)
        docker run -d --restart unless-stopped --name "$CONTAINER" "${RUN_ARGS[@]}" "$IMAGE" server "$@" || exit 1
        echo "Ready at \"Whisper server listening on ...\": bash $0 $BOARD logs"
        ;;
    stop)
        docker rm -f "$CONTAINER"
        ;;
    logs)
        exec docker logs -f "$CONTAINER"
        ;;
    status)
        docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Names}} {{.Image}} {{.Status}}'
        health || { echo "no answer at $URL" >&2; exit 1; }
        ;;
    test)
        WAV=${1:-$(ls "$ROOT_DIR"/audio_files/*.wav | head -n 1)}
        health || { echo "no answer at $URL" >&2; exit 1; }
        for i in 1 2 3; do
            curl -s -X POST -H 'Content-Type: audio/wav' --data-binary @"$WAV" -w "  (%{time_total} s)\n" "$URL/transcribe"
        done
        ;;
    shell)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" shell
        ;;
    *)
        sed -n 4,13p "$0"; exit 1
        ;;
esac

#!/bin/bash
# Builds and runs the Piper TTS server in Docker (no ROS). ari_servers is mounted as /workspace.
#
# Usage: bash run.bash <build|server|start|stop|logs|status|test|shell> [args]
#   build   builds the image ari_tts_server (base: ari_whisper_server:spark)
#   server  runs the server in the foreground: [tts_server args]
#   start   runs it in the background (container tts_server, it restarts with the machine): [tts_server args]
#   stop    stops and removes the container tts_server
#   logs    follows the log
#   status  container state and GET /health
#   test    one request, the audio in /tmp/tts_test.wav: [text] [voice]
#   shell   shell inside the image
#
# Configuration: section tts of config/servers.yaml. Variables exported here override the file: TTS_<FIELD>
# (TTS_USE_CUDA, TTS_VOICES, ...), ARI_EXECUTION_LOCK. status and test use TTS_PORT, default 8020.

COMMAND=$1
shift || { sed -n 4,13p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=ari_tts_server
CONTAINER=tts_server
URL=http://localhost:${TTS_PORT:-8020}

RUN_ARGS=(--gpus=all --network=host -v "$ROOT_DIR":/workspace)
# only the variables set here: otherwise defaults would hide config/servers.yaml
for variable in $(compgen -e | grep -E '^(TTS_|ARI_EXECUTION_LOCK$|ARI_SERVERS_CONFIG$)'); do
    RUN_ARGS+=(-e "$variable=${!variable}")
done

health() {
    curl -sf -m 5 "$URL/health" && echo
}

case "$COMMAND" in
    build)
        exec docker build -t "$IMAGE" "$SCRIPT_DIR"
        ;;
    server)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server "$@"
        ;;
    start)
        docker run -d --restart unless-stopped --name "$CONTAINER" "${RUN_ARGS[@]}" "$IMAGE" server "$@" || exit 1
        echo "Ready at \"tts_server listening on ...\": bash $0 status"
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
        health || { echo "no answer at $URL" >&2; exit 1; }
        TEXT=${1:-Che bella domanda. Lasciami pensare}
        VOICE=${2:+, \"voice\": \"$2\"}
        curl -s -m 30 -X POST -H 'Content-Type: application/json' -D - -o /tmp/tts_test.wav \
            -w "(%{http_code}, %{size_download} bytes, %{time_total} s) -> /tmp/tts_test.wav\n" \
            -d "{\"text\": \"$TEXT\"$VOICE}" "$URL/synthesize" | grep -E '^X-|^\('
        ;;
    shell)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" shell
        ;;
    *)
        sed -n 4,13p "$0"; exit 1
        ;;
esac

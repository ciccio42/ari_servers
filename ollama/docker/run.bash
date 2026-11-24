#!/bin/bash
# Ollama in Docker: the local models of the RAGs (rag/<name>), used by the LLM server (backend ollama) for the
# answers and by the RAG servers for their planners. One server for all the RAGs, models in models/ollama.
#
# Usage: bash run.bash <start|stop|status|logs|pull|list> [model]
#   start   runs the container ollama_server in the background (it restarts with the machine)
#   stop    stops and removes it
#   status  container state and the models present
#   logs    follows the log
#   pull    downloads a model: pull <model> (default: the generator of the department_office RAG)
#   list    models present
#
# Configuration: section ollama of config/servers.yaml (url, num_ctx, keep_alive, think). OLLAMA_IMAGE: another image.

COMMAND=$1
shift || { sed -n 5,12p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=${OLLAMA_IMAGE:-ollama/ollama:0.40.16}
CONTAINER=ollama_server
CONFIG=$ROOT_DIR/${ARI_SERVERS_CONFIG:-config/servers.yaml}

# address of the configuration (section ollama, key url), default http://localhost:11500
URL=$(python3 -c 'import sys, yaml; print(((yaml.safe_load(open(sys.argv[1])) or {}).get("ollama") or {}).get("url", "http://localhost:11500"))' "$CONFIG")
HOST_PORT=${URL#*://}
PORT=${HOST_PORT##*:}

case "$COMMAND" in
    start)
        mkdir -p "$ROOT_DIR/models/ollama"
        # 127.0.0.1: only the servers of this machine (network host). NUM_PARALLEL 1: one request at a time, as the
        # execution lock of the servers. CONTEXT_LENGTH: default context, the requests of the RAG give their own
        docker run -d --restart unless-stopped --name "$CONTAINER" --gpus=all --network=host \
            -e OLLAMA_HOST="127.0.0.1:$PORT" -e OLLAMA_MODELS=/models -e OLLAMA_NUM_PARALLEL=1 \
            -e OLLAMA_KEEP_ALIVE=2h -e OLLAMA_CONTEXT_LENGTH=16384 \
            -v "$ROOT_DIR/models/ollama":/models "$IMAGE" || exit 1
        echo "Ollama at $URL: bash $0 status"
        ;;
    stop)
        docker rm -f "$CONTAINER"
        ;;
    status)
        docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Names}} {{.Image}} {{.Status}}'
        curl -sf -m 5 "$URL/api/version" && echo || { echo "no answer at $URL" >&2; exit 1; }
        docker exec -e OLLAMA_HOST="127.0.0.1:$PORT" "$CONTAINER" ollama list
        ;;
    logs)
        exec docker logs -f "$CONTAINER"
        ;;
    pull)
        MODEL=${1:-qwen3.6:35b-a3b-q8_0}
        TTY=(); [ -t 1 ] && TTY=(-it)
        exec docker exec "${TTY[@]}" -e OLLAMA_HOST="127.0.0.1:$PORT" "$CONTAINER" ollama pull "$MODEL"
        ;;
    list)
        exec docker exec -e OLLAMA_HOST="127.0.0.1:$PORT" "$CONTAINER" ollama list
        ;;
    *)
        sed -n 5,12p "$0"; exit 1
        ;;
esac

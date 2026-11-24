#!/bin/bash
# Builds and runs the orchestrator of the models in Docker (no ROS, no GPU): one pair LLM + RAG at a time.
#
# Usage: bash run.bash <build|server|start|stop|logs|status|load|unload|shell> [args]
#   build   builds the image ari_orchestrator
#   server  runs the orchestrator in the foreground: [orchestrator_server args]
#   start   runs it in the background (container orchestrator_server, it restarts with the machine)
#   stop    stops and removes the container
#   logs    follows the log
#   status  container state and GET /status
#   load    loads a pair: [rag=<name>] [llm=<backend>:<model>], e.g. load rag=high_level_unisa, load llm=ollama:qwen3.6:35b-a3b
#   unload  stops the RAG containers and unloads the models of Ollama
#   shell   shell inside the image
#
# Configuration: section orchestrator of config/servers.yaml; variables exported here override it: ORCHESTRATOR_<FIELD>,
# ARI_SERVERS_CONFIG. status, load and unload use ORCHESTRATOR_PORT, default 8040.
# The container has the socket of Docker (it starts and stops the RAG containers and Ollama) and ari_servers at its
# path of the host, so the run.bash of the servers mount the same folders.

COMMAND=$1
shift || { sed -n 4,15p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=ari_orchestrator
CONTAINER=orchestrator_server
URL=http://localhost:${ORCHESTRATOR_PORT:-8040}

RUN_ARGS=(--network=host -v /var/run/docker.sock:/var/run/docker.sock -v "$ROOT_DIR":"$ROOT_DIR"
          -e WORKSPACE="$ROOT_DIR")
for variable in $(compgen -e | grep -E '^(ORCHESTRATOR_|ARI_SERVERS_CONFIG$)'); do
    RUN_ARGS+=(-e "$variable=${!variable}")
done

pretty() { python3 -m json.tool 2> /dev/null || cat; }

case "$COMMAND" in
    build)
        exec docker build -t "$IMAGE" "$SCRIPT_DIR"
        ;;
    server)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server "$@"
        ;;
    start)
        docker run -d --restart unless-stopped --name "$CONTAINER" "${RUN_ARGS[@]}" "$IMAGE" server "$@" || exit 1
        echo "Ready at \"orchestrator listening on ...\": bash $0 status"
        ;;
    stop)
        docker rm -f "$CONTAINER"
        ;;
    logs)
        exec docker logs -f "$CONTAINER"
        ;;
    status)
        docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Names}} {{.Image}} {{.Status}}'
        OUT=$(curl -sf -m 10 "$URL/status") || { echo "no answer at $URL" >&2; exit 1; }
        pretty <<< "$OUT"
        ;;
    load)
        BODY=$(python3 -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:])))' "$@")
        curl -s -m 1800 -X POST -H 'Content-Type: application/json' -d "$BODY" -w "\n  (%{http_code}, %{time_total} s)\n" "$URL/load"
        ;;
    unload)
        curl -s -m 300 -X POST -H 'Content-Type: application/json' -d '{}' "$URL/unload" | pretty
        ;;
    shell)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" shell
        ;;
    *)
        sed -n 4,15p "$0"; exit 1
        ;;
esac

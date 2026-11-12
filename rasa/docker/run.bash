#!/bin/bash
# Builds and runs the RASA server in Docker (no ROS): the 3 bots, the action server of the social bot and the proxy
# with the execution lock. ari_servers is mounted as /workspace.
#
# Usage: bash run.bash <build|server|start|stop|logs|status|test|train|shell> [args]
#   build   builds the image ari_rasa_server
#   server  runs it in the foreground
#   start   runs it in the background (container rasa_server, it restarts with the machine)
#   stop    stops and removes the container rasa_server
#   logs    follows the log
#   status  container state and GET /health of every port of the proxy
#   test    one message to every bot through the proxy
#   train   [<bot>|all] [it|en|all]: rasa train into models/rasa/<bot>/<lang>
#   shell   shell inside the image
#
# Configuration: config/rasa.env (versioned): language, address of the LLM server, flags of the demo; RASA_ENV_FILE:
# another file. The answers of the social bot come from the LLM server (llm/docker/run.bash start). Variables exported
# in the shell override the file: RASA_<FIELD> (RASA_ROUTES, RASA_EXCLUSIVE, ...: the proxy, otherwise from
# config/servers.yaml), RASA_BOTS, ACTIONS_PORT, DEMO_LANGUAGE, ARI_EXECUTION_LOCK, MAX_CHAR, DEBUG,
# KNOWLEDGE_GRAPH_RUNNING, LLM_URL, LLM_TIMEOUT.
# status and test use the ports 5002 5003 5004 (RASA_PORTS to change them).

COMMAND=$1
shift || { sed -n 4,14p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=ari_rasa_server
CONTAINER=rasa_server
PORTS=${RASA_PORTS:-5002 5003 5004}

RUN_ARGS=(--network=host -v "$ROOT_DIR":/workspace)
# only the variables set here: otherwise defaults would hide config/servers.yaml
for variable in $(compgen -e | grep -E '^(RASA_|ACTIONS_PORT$|DEMO_LANGUAGE$|ARI_EXECUTION_LOCK$|MAX_CHAR$|DEBUG$|KNOWLEDGE_GRAPH_RUNNING$|LLM_URL$|LLM_TIMEOUT$)'); do
    case "$variable" in RASA_ENV_FILE|RASA_PORTS) ;; *) RUN_ARGS+=(-e "$variable=${!variable}") ;; esac
done
ENV_FILE=${RASA_ENV_FILE:-$ROOT_DIR/config/rasa.env}
[ -f "$ENV_FILE" ] || { echo "$ENV_FILE not found" >&2; exit 1; }
RUN_ARGS+=(--env-file "$ENV_FILE")

# before starting the server: docker --env-file keeps the quotes in the values. The LLM server: a warning only, the
# bots and the games work without it
check_config() {
    if grep -n -E "^[A-Za-z_]+=[\"']" "$ENV_FILE"; then
        echo "$ENV_FILE: write the values without quotes (lines above)" >&2; exit 1
    fi
    local url=${LLM_URL:-$(sed -n 's/^LLM_URL=//p' "$ENV_FILE")}
    curl -sf -m 3 "${url:-http://localhost:8010}/health" > /dev/null \
        || echo "warning: no LLM server at ${url:-http://localhost:8010}: the social bot will answer with its courtesy phrase" >&2
}

health() {
    local ok=0
    for port in $PORTS; do
        curl -sf -m 5 "http://localhost:$port/health" && echo || { echo "no answer at :$port" >&2; ok=1; }
    done
    return $ok
}

case "$COMMAND" in
    build)
        exec docker build -t "$IMAGE" "$SCRIPT_DIR"
        ;;
    server)
        check_config
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server
        ;;
    start)
        check_config
        docker run -d --restart unless-stopped --name "$CONTAINER" "${RUN_ARGS[@]}" "$IMAGE" server || exit 1
        echo "The bots load in about a minute: bash $0 status"
        ;;
    stop)
        docker rm -f "$CONTAINER"
        ;;
    logs)
        exec docker logs -f "$CONTAINER"
        ;;
    status)
        docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Names}} {{.Image}} {{.Status}}'
        health
        ;;
    test)
        health || exit 1
        for port in $PORTS; do
            echo "== :$port"
            curl -s -m 60 -X POST -H 'Content-Type: application/json' -w "  (%{time_total} s)\n" \
                -d "{\"sender\": \"test\", \"message\": \"${1:-ciao}\"}" "http://localhost:$port/webhooks/rest/webhook"
        done
        ;;
    train)
        exec docker run --rm "${RUN_ARGS[@]}" "$IMAGE" train "$@"
        ;;
    shell)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" shell
        ;;
    *)
        sed -n 4,14p "$0"; exit 1
        ;;
esac

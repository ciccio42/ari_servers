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
# Passed to the container only if set here, so they override config/servers.yaml: RASA_<FIELD> (RASA_ROUTES,
# RASA_EXCLUSIVE, ...), RASA_BOTS, ACTIONS_PORT, DEMO_LANGUAGE (default Italian), ARI_EXECUTION_LOCK, MAX_CHAR, DEBUG,
# KNOWLEDGE_GRAPH_RUNNING, OPENAI_* (OPENAI_API_KEY is needed by the actions of the social bot: never in the code).
# RASA_ENV_FILE: a file of variables (e.g. OPENAI_API_KEY=...) passed with --env-file, not versioned.
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
for variable in $(compgen -e | grep -E '^(RASA_|ACTIONS_PORT$|DEMO_LANGUAGE$|ARI_EXECUTION_LOCK$|MAX_CHAR$|DEBUG$|KNOWLEDGE_GRAPH_RUNNING$|OPENAI_)'); do
    case "$variable" in RASA_ENV_FILE|RASA_PORTS) ;; *) RUN_ARGS+=(-e "$variable=${!variable}") ;; esac
done
[ -n "$RASA_ENV_FILE" ] && RUN_ARGS+=(--env-file "$RASA_ENV_FILE")

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
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server
        ;;
    start)
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

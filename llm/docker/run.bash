#!/bin/bash
# Builds and runs the LLM (+RAG) server in Docker (no ROS). ari_servers is mounted as /workspace.
#
# Usage: bash run.bash <build|server|start|stop|logs|status|test|shell> [args]
#   build   builds the image ari_llm_server (base: ari_whisper_server:spark)
#   server  runs the server in the foreground: [llm_server args]
#   start   runs it in the background (container llm_server, it restarts with the machine): [llm_server args]
#   stop    stops and removes the container llm_server
#   logs    follows the log
#   status  container state and GET /health
#   test    one request: [message] (default: a question of the test plan)
#   shell   shell inside the image
#
# Configuration: sections llm and openai of config/servers.yaml; config/secrets.env (not versioned) for
# OPENAI_API_KEY. Variables exported here override the file: LLM_<FIELD> (LLM_BACKEND, LLM_MODEL, LLM_RAG, ...),
# OPENAI_<FIELD>, ARI_EXECUTION_LOCK. status and test use LLM_PORT, default 8010.

COMMAND=$1
shift || { sed -n 4,13p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=ari_llm_server
CONTAINER=llm_server
URL=http://localhost:${LLM_PORT:-8010}
SECRETS_FILE=$ROOT_DIR/config/secrets.env

RUN_ARGS=(--gpus=all --network=host -e HF_HOME=/workspace/models/llm/.hf_cache -v "$ROOT_DIR":/workspace)
# only the variables set here: otherwise defaults would hide config/servers.yaml
for variable in $(compgen -e | grep -E '^(LLM_|OPENAI_|ARI_EXECUTION_LOCK$)'); do
    RUN_ARGS+=(-e "$variable=${!variable}")
done
[ -f "$SECRETS_FILE" ] && RUN_ARGS+=(--env-file "$SECRETS_FILE")

# the ChatGPT backend stops at the start without the key; docker --env-file keeps the quotes in the values
check_config() {
    if [ -f "$SECRETS_FILE" ] && grep -n -E "^[A-Za-z_]+=[\"']" "$SECRETS_FILE"; then
        echo "$SECRETS_FILE: write the values without quotes (lines above)" >&2; exit 1
    fi
    local backend=${LLM_BACKEND:-$(sed -n '/^llm:/,/^[^ #]/s/^  backend: *\([a-z_]*\).*/\1/p' "$ROOT_DIR/config/servers.yaml")}
    if [ "${backend:-openai}" = openai ] && [ -z "$OPENAI_API_KEY" ] && ! grep -q -s -E '^OPENAI_API_KEY=.+' "$SECRETS_FILE"; then
        echo "OPENAI_API_KEY not set: echo 'OPENAI_API_KEY=sk-...' > config/secrets.env && chmod 600 config/secrets.env" >&2
        exit 1
    fi
}

health() {
    curl -sf -m 5 "$URL/health" && echo
}

case "$COMMAND" in
    build)
        exec docker build -t "$IMAGE" "$SCRIPT_DIR"
        ;;
    server)
        check_config
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server "$@"
        ;;
    start)
        check_config
        docker run -d --restart unless-stopped --name "$CONTAINER" "${RUN_ARGS[@]}" "$IMAGE" server "$@" || exit 1
        echo "Ready at \"llm_server listening on ...\": bash $0 status"
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
        MESSAGE=${1:-quali sono le soglie di emoglobina glicata che indicano un rischio di diabete?}
        curl -s -m 120 -X POST -H 'Content-Type: application/json' -w "  (%{http_code}, %{time_total} s)\n" \
            -d "{\"conversation_id\": \"run.bash-test\", \"message\": \"$MESSAGE\"}" "$URL/chat"
        ;;
    shell)
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" shell
        ;;
    *)
        sed -n 4,13p "$0"; exit 1
        ;;
esac

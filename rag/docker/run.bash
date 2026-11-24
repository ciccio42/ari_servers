#!/bin/bash
# Builds and runs the RAG servers in Docker (no ROS): one container rag_<name> per RAG, from the image ari_rag_server
# (one conda environment per RAG). ari_servers is mounted as /workspace.
#
# Usage: bash run.bash <build|server|start|stop|logs|status|test|shell> [name ...] [-- args]
#   build   builds the image ari_rag_server with the conda environments of every rag/<name>/environment.yml
#   server  runs one RAG in the foreground: <name> [-- <name>_rag args]
#   start   runs the RAGs in the background (containers rag_<name>, they restart with the machine)
#   stop    stops and removes their containers
#   logs    follows the log of one RAG: <name>
#   status  container state and GET /health of the RAGs
#   test    one retrieval: <name> [question]
#   shell   shell in the conda environment of one RAG: <name>
# Without names: the RAGs of rag_servers in config/servers.yaml.
#
# Configuration: section rag_<name> of config/servers.yaml (port, knowledge base, models); variables exported here
# override it: RAG_<NAME>_<FIELD> (e.g. RAG_DEPARTMENT_OFFICE_TOP_K), ARI_EXECUTION_LOCK, ARI_SERVERS_CONFIG.
# The models of the retrieval (BGE-M3, reranker) are downloaded at the first start in models/rag/.hf_cache.

COMMAND=$1
shift || { sed -n 4,15p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
IMAGE=ari_rag_server
CONFIG=$ROOT_DIR/${ARI_SERVERS_CONFIG:-config/servers.yaml}

NAMES=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do NAMES+=("$1"); shift; done
[ "$1" = -- ] && shift   # the rest: arguments of the server

# rag_servers of the configuration, and the port of every RAG (section rag_<name>, default 8030)
config_value() {  # config_value rag_servers | port <name>
    python3 - "$CONFIG" "$@" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
if sys.argv[2] == "rag_servers":
    print(" ".join(str(n) for n in (c.get("rag_servers") or [])))
else:
    print((c.get(f"rag_{sys.argv[3]}") or {}).get("port", 8030))
PY
}
[ ${#NAMES[@]} -eq 0 ] && read -r -a NAMES <<< "$(config_value rag_servers)"

check_name() {
    [ -f "$ROOT_DIR/rag/$1/environment.yml" ] || { echo "unknown RAG $1: rag/$1/environment.yml not found" >&2; exit 1; }
}

one_name() {
    [ ${#NAMES[@]} -eq 1 ] || { echo "$COMMAND needs one RAG: $(ls "$ROOT_DIR/rag" | grep -v docker | tr '\n' ' ')" >&2; exit 1; }
    check_name "${NAMES[0]}"
}

run_args() {  # run_args <name>
    local prefix variable
    prefix=RAG_$(tr '[:lower:]' '[:upper:]' <<< "$1")_
    RUN_ARGS=(--gpus=all --network=host -e HF_HOME=/workspace/models/rag/.hf_cache -v "$ROOT_DIR":/workspace)
    # only the variables set here: otherwise defaults would hide config/servers.yaml
    for variable in $(compgen -e | grep -E "^(${prefix}|ARI_EXECUTION_LOCK$|ARI_SERVERS_CONFIG$|HF_HUB_OFFLINE$)"); do
        RUN_ARGS+=(-e "$variable=${!variable}")
    done
}

health() {  # health <name>
    curl -sf -m 5 "http://localhost:$(config_value port "$1")/health" && echo
}

case "$COMMAND" in
    build)
        exec docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$ROOT_DIR/rag"
        ;;
    server)
        one_name
        run_args "${NAMES[0]}"
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" server "${NAMES[0]}" "$@"
        ;;
    start)
        for name in "${NAMES[@]}"; do
            check_name "$name"
            run_args "$name"
            docker run -d --restart unless-stopped --name "rag_$name" "${RUN_ARGS[@]}" "$IMAGE" server "$name" "$@" \
                || exit 1
            echo "rag_$name: ready at \"rag_$name listening on ...\": bash $0 status $name"
        done
        ;;
    stop)
        for name in "${NAMES[@]}"; do docker rm -f "rag_$name"; done
        ;;
    logs)
        one_name
        exec docker logs -f "rag_${NAMES[0]}"
        ;;
    status)
        failed=0
        for name in "${NAMES[@]}"; do
            docker ps -a --filter "name=^rag_${name}$" --format '{{.Names}} {{.Image}} {{.Status}}'
            health "$name" || { echo "no answer from rag_$name" >&2; failed=1; }
        done
        exit $failed
        ;;
    test)
        [ ${#NAMES[@]} -ge 1 ] || { echo "test needs one RAG" >&2; exit 1; }
        name=${NAMES[0]}
        QUESTION=${NAMES[1]:-${1:-Quali studenti hanno chiesto il passaggio di corso nel verbale del 29/09/2025?}}
        health "$name" > /dev/null || { echo "no answer from rag_$name" >&2; exit 1; }
        curl -s -m 300 -X POST -H 'Content-Type: application/json' -w "\n  (%{http_code}, %{time_total} s)\n" \
            -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$QUESTION")" \
            "http://localhost:$(config_value port "$name")/retrieve" \
            | python3 -c 'import json,sys
raw = sys.stdin.read(); body, _, tail = raw.rpartition("\n  (")
try:
    r = json.loads(body)
    print("filters:", r.get("filters"), "| registers:", r.get("registers") or "-", "| chunks:", len(r.get("chunks", [])),
          "| context:", len(r.get("user", "")), "characters | retrieve_s:", r.get("retrieve_s"))
    for c in r.get("chunks", [])[:8]:
        print("  %.3f  %s" % (c["score"], c["source"][:110]))
    print("answer:", r.get("answer") or "(generated by the LLM server)")
except ValueError:
    print(body)
print("  (" + tail)'
        ;;
    shell)
        one_name
        run_args "${NAMES[0]}"
        exec docker run --rm -it "${RUN_ARGS[@]}" "$IMAGE" shell "${NAMES[0]}"
        ;;
    *)
        sed -n 4,15p "$0"; exit 1
        ;;
esac

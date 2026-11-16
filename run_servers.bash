#!/bin/bash
# Starts, stops and checks the servers of the DGX Spark (architecture: ari_demo_arch.png), in Docker:
# Whisper (speech to text), LLM (+RAG) (answers), TTS (Piper, audio of the answers) and RASA (dialogue). Each one runs with its own run.bash; this script
# keeps the order (the RASA server asks the LLM server) and one configuration for all of them.
#
# Usage: bash run_servers.bash [-c <servers.yaml>] <start|stop|restart|status|logs|build> [whisper|llm|tts|rasa ...]
#   start    starts the servers not running yet and waits until they answer (default: all four)
#   stop     stops them
#   restart  stop + start: needed after a change of the configuration
#   status   container state and GET /health of every server
#   logs     follows the log of one server: logs <whisper|llm|tts|rasa>
#   build    builds the images (once; the Whisper image first, the LLM and TTS images are built on it)
#
# Configuration (paths relative to ari_servers):
#   config/servers.yaml   ports, models, RAG on/off, OpenAI parameters, execution lock of all the servers.
#                         -c <file> (or ARI_SERVERS_CONFIG) uses another file, inside ari_servers
#   config/rasa.env       language, address of the LLM server, flags of the demo (RASA_ENV_FILE: another file)
#   config/secrets.env    OPENAI_API_KEY=... (not versioned): needed by the LLM server with the ChatGPT backend
# Variables exported in the shell override the files, e.g. LLM_RAG=true bash run_servers.bash restart llm.
# BOARD: board of the Whisper image (default spark).

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR" || exit 1

usage() { sed -n 6,13p "$0"; exit 1; }

if [ "$1" = -c ] || [ "$1" = --config ]; then
    [ -n "$2" ] || usage
    ARI_SERVERS_CONFIG=$2
    shift 2
fi
COMMAND=$1
[ -n "$COMMAND" ] && shift || usage
SERVERS=("$@")
[ ${#SERVERS[@]} -eq 0 ] && SERVERS=(whisper llm tts rasa)
[ "$COMMAND" = stop ] && [ $# -eq 0 ] && SERVERS=(rasa tts llm whisper)   # reverse order

# the configuration file: inside ari_servers, so the containers see it in /workspace with the same relative path
CONFIG=${ARI_SERVERS_CONFIG:-config/servers.yaml}
CONFIG_PATH=$(realpath -m "$CONFIG")
case "$CONFIG_PATH" in
    "$ROOT_DIR"/*) export ARI_SERVERS_CONFIG=${CONFIG_PATH#"$ROOT_DIR"/} ;;
    *) echo "$CONFIG: the configuration must be inside $ROOT_DIR (it is mounted in the containers)" >&2; exit 1 ;;
esac
[ -f "$ARI_SERVERS_CONFIG" ] || { echo "$ARI_SERVERS_CONFIG not found" >&2; exit 1; }

# ports of the configuration, for status and for the wait (the servers read them from the file anyway)
eval "$(python3 - "$ARI_SERVERS_CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
routes = str((c.get("rasa") or {}).get("routes", "5002=,5003=,5004="))
print(f"WHISPER_PORT_CFG={(c.get('whisper') or {}).get('port', 8000)}")
print(f"LLM_PORT_CFG={(c.get('llm') or {}).get('port', 8010)}")
print(f"TTS_PORT_CFG={(c.get('tts') or {}).get('port', 8020)}")
print("RASA_PORTS_CFG='" + " ".join(r.split("=")[0].strip() for r in routes.split(",") if r.strip()) + "'")
PY
)" || { echo "$ARI_SERVERS_CONFIG: cannot read it" >&2; exit 1; }
export WHISPER_PORT=${WHISPER_PORT:-$WHISPER_PORT_CFG} LLM_PORT=${LLM_PORT:-$LLM_PORT_CFG} TTS_PORT=${TTS_PORT:-$TTS_PORT_CFG}
export RASA_PORTS=${RASA_PORTS:-$RASA_PORTS_CFG}
BOARD=${BOARD:-spark}

run() {  # run <server> <command> [args]: the run.bash of the server
    case "$1" in
        whisper) bash whisper/docker/run.bash "$BOARD" "${@:2}" ;;
        llm)     bash llm/docker/run.bash "${@:2}" ;;
        tts)     bash tts/docker/run.bash "${@:2}" ;;
        rasa)    bash rasa/docker/run.bash "${@:2}" ;;
        *) echo "unknown server $1: whisper, llm, tts, rasa" >&2; return 1 ;;
    esac
}

container() {
    case "$1" in whisper) echo whisper_server ;; llm) echo llm_server ;; tts) echo tts_server ;; rasa) echo rasa_server ;; esac
}

image() {
    case "$1" in whisper) echo "ari_whisper_server:$BOARD" ;; llm) echo ari_llm_server ;; tts) echo ari_tts_server ;; rasa) echo ari_rasa_server ;; esac
}

ready() {  # the server answers; for rasa also every bot behind the proxy
    local out
    out=$(run "$1" status 2> /dev/null) || return 1
    [ "$1" != rasa ] || [ "$(grep -c '"upstream_ok": true' <<< "$out")" -eq "$(wc -w <<< "$RASA_PORTS")" ]
}

wait_ready() {  # wait_ready <server> <seconds>
    local start=$SECONDS
    while [ $((SECONDS - start)) -lt "$2" ]; do
        if ready "$1"; then
            echo "  $1 ready after $((SECONDS - start)) s"
            return 0
        fi
        if [ -z "$(docker ps -q --filter "name=^$(container "$1")$")" ]; then
            echo "  $1 stopped: bash run_servers.bash logs $1" >&2
            return 1
        fi
        sleep 3
    done
    echo "  $1 not ready after $2 s: bash run_servers.bash logs $1" >&2
    return 1
}

start_one() {
    local name state current
    name=$(container "$1")
    state=$(docker inspect -f '{{.State.Status}}' "$name" 2> /dev/null)
    if [ -n "$state" ]; then
        current=$(docker inspect -f '{{.Config.Image}}' "$name")
        if [ "$current" != "$(image "$1")" ]; then
            echo "  container $name exists with another image ($current): stop it first (docker stop $name)" >&2
            return 1
        fi
        if [ "$state" = running ]; then
            echo "  $1 already running (restart to apply a new configuration)"
            return 0
        fi
        docker rm "$name" > /dev/null  # stopped: start it again with the current configuration
    fi
    run "$1" start > /dev/null || return 1
    # first start of Whisper: download and export of the model (minutes); the bots of Rasa: about a minute
    case "$1" in whisper) wait_ready whisper 900 ;; llm) wait_ready llm 120 ;; tts) wait_ready tts 60 ;; rasa) wait_ready rasa 180 ;; esac
}

echo "Configuration: $ARI_SERVERS_CONFIG, ${RASA_ENV_FILE:-config/rasa.env}, config/secrets.env"
case "$COMMAND" in
    start)
        for server in "${SERVERS[@]}"; do
            echo "== $server"
            start_one "$server" || exit 1
        done
        ;;
    stop)
        for server in "${SERVERS[@]}"; do
            echo "== $server"
            run "$server" stop > /dev/null 2>&1 && echo "  stopped" || echo "  not running"
        done
        ;;
    restart)
        [ $# -eq 0 ] && STOP=(rasa tts llm whisper) || STOP=("${SERVERS[@]}")
        bash "$0" -c "$ARI_SERVERS_CONFIG" stop "${STOP[@]}" > /dev/null
        exec bash "$0" -c "$ARI_SERVERS_CONFIG" start "${SERVERS[@]}"
        ;;
    status)
        failed=0
        for server in "${SERVERS[@]}"; do
            echo "== $server"
            run "$server" status 2>&1 | sed 's/^/  /'
            ready "$server" || { echo "  NOT READY"; failed=1; }
        done
        exit $failed
        ;;
    logs)
        [ $# -eq 1 ] || { echo "logs needs one server: whisper, llm, tts or rasa" >&2; exit 1; }
        exec docker logs -f "$(container "$1")"
        ;;
    build)
        for server in "${SERVERS[@]}"; do
            echo "== $server"
            run "$server" build || exit 1
        done
        ;;
    *)
        usage
        ;;
esac

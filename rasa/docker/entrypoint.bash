#!/bin/bash
# Entrypoint of the ari_rasa_server image. ari_servers is mounted as /workspace: bots and actions are read from
# /workspace/rasa/bots, the trained models from /workspace/models/rasa/<bot>/<it|en>, the latency log goes to
# /workspace/latency_logs. No ROS.
#
# Usage: entrypoint.bash [server | train [<bot>|all] [it|en|all] | shell | <command>]
#   server  (default) action server of the social bot (5055), the bots on their internal ports and the proxy on the
#           ports of the clients (rasa.routes of config/servers.yaml). If one of them stops, the container stops.
#   train   rasa train of the bots from rasa/bots/<bot>/config/models/<lang> to models/rasa/<bot>/<lang>
#
# DEMO_LANGUAGE (Italian, English) picks the models. RASA_BOTS: <bot>:<internal port> of the bots, they must match
# the upstreams of rasa.routes. OPENAI_API_KEY and OPENAI_* for the actions of the social bot.
set -e

export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/rasa${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export DEMO_LANGUAGE=${DEMO_LANGUAGE:-Italian}  # also read by the actions (text_manager.py)
BOTS_DIR=$WORKSPACE/rasa/bots
MODELS_DIR=$WORKSPACE/models/rasa
RASA_BOTS=${RASA_BOTS:-social:5012 rock_paper_scissor:5013 connect_four:5014}
ACTIONS_PORT=${ACTIONS_PORT:-5055}  # action_endpoint of bots/social/config/endpoints.yml

lang_code() {
    case "${1,,}" in
        italian|it) echo it ;;
        english|en) echo en ;;
        *) echo "DEMO_LANGUAGE must be Italian or English, got '$1'" >&2; return 1 ;;
    esac
}

train() {
    local bot=$1 lang=$2 data=$BOTS_DIR/$1/config/models/$2
    echo "===== train $bot $lang"
    (cd "$BOTS_DIR/$bot" && rasa train --config "$data/config.yml" --domain "$data/domain.yml" --data "$data" \
        --out "$MODELS_DIR/$bot/$lang")
}

case "${1:-server}" in
    server)
        LANG_CODE=$(lang_code "${DEMO_LANGUAGE:-Italian}")
        (cd "$BOTS_DIR/social" && exec rasa run actions --actions actions --port "$ACTIONS_PORT") &
        for entry in $RASA_BOTS; do
            bot=${entry%%:*} port=${entry##*:}
            ls "$MODELS_DIR/$bot/$LANG_CODE"/*.tar.gz > /dev/null || { echo "no model in $MODELS_DIR/$bot/$LANG_CODE" >&2; exit 1; }
            (cd "$BOTS_DIR/$bot" && exec rasa run -m "$MODELS_DIR/$bot/$LANG_CODE" --endpoints config/endpoints.yml \
                --credentials config/credentials.yml --enable-api --port "$port") &
        done
        (cd "$WORKSPACE" && exec python3 -m rasa_proxy) &
        trap 'kill $(jobs -p) 2> /dev/null' EXIT
        wait -n
        echo "a process of the RASA server stopped: stopping the container" >&2
        exit 1
        ;;
    train)
        BOTS=${2:-all} LANGS=${3:-all}
        [ "$BOTS" = all ] && BOTS=$(for entry in $RASA_BOTS; do echo "${entry%%:*}"; done)
        [ "$LANGS" = all ] && LANGS="it en"
        for bot in $BOTS; do for lang in $LANGS; do train "$bot" "$lang"; done; done
        ;;
    shell)
        exec bash
        ;;
    *)
        exec "$@"
        ;;
esac

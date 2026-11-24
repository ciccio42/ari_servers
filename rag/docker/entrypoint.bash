#!/bin/bash
# Entrypoint of the ari_rag_server image. ari_servers is mounted as /workspace: the code of the RAG is read from
# /workspace/rag/<name> at every start (a change needs only a restart of the container), the configuration from
# /workspace/config. Every RAG runs in its conda environment rag_<name>.
#
# Usage: entrypoint.bash [server <name> [<name>_rag args] | shell [<name>] | envs | <command>]
set -e

case "${1:-}" in
    server)
        NAME=$2
        [ -n "$NAME" ] || { echo "server needs the name of the RAG (rag/<name>)" >&2; exit 1; }
        shift 2
        [ -d "$WORKSPACE/rag/$NAME" ] || { echo "$WORKSPACE/rag/$NAME not found" >&2; exit 1; }
        export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/rag/$NAME${PYTHONPATH:+:$PYTHONPATH}
        cd "$WORKSPACE"
        exec conda run --no-capture-output -n "rag_$NAME" python -m "${NAME}_rag" "$@"
        ;;
    shell)
        if [ -n "$2" ]; then
            export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/rag/$2
            source /opt/conda/etc/profile.d/conda.sh
            conda activate "rag_$2"
        fi
        exec bash
        ;;
    envs)
        exec conda env list
        ;;
    "")
        echo "usage: server <name> | shell [<name>] | envs | <command>" >&2
        exit 1
        ;;
    *)
        exec "$@"
        ;;
esac

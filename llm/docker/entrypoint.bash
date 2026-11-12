#!/bin/bash
# Entrypoint of the ari_llm_server image. ari_servers is mounted as /workspace: the code is read from /workspace/llm
# at every start (a change needs only a restart of the container), the configuration from /workspace/config.
#
# Usage: entrypoint.bash [server [llm_server args] | shell | <command>]
set -e

export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/llm${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

case "${1:-server}" in
    server)
        shift || true
        cd "$WORKSPACE"
        exec python3 -m llm_server "$@"
        ;;
    shell)
        exec bash
        ;;
    *)
        exec "$@"
        ;;
esac

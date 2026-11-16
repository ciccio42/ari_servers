#!/bin/bash
# Entrypoint of the ari_tts_server image. ari_servers is mounted as /workspace: the code is read from /workspace/tts
# at every start (a change needs only a restart of the container), the configuration from /workspace/config.
#
# Usage: entrypoint.bash [server [tts_server args] | shell | <command>]
set -e

export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/tts${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

case "${1:-server}" in
    server)
        shift || true
        cd "$WORKSPACE"
        exec python3 -m tts_server "$@"
        ;;
    shell)
        exec bash
        ;;
    *)
        exec "$@"
        ;;
esac

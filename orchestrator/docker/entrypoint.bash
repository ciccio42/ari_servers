#!/bin/bash
# Entrypoint of the ari_orchestrator image. ari_servers is mounted at its path of the host ($WORKSPACE), so the run.bash
# of the servers started from here mount the same folders as from the host; the code is read from there at every start.
#
# Usage: entrypoint.bash [server [orchestrator_server args] | shell | <command>]
set -e
export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/orchestrator${PYTHONPATH:+:$PYTHONPATH}

case "${1:-server}" in
    server)
        shift || true
        cd "$WORKSPACE"
        exec python3 -m orchestrator_server "$@"
        ;;
    shell)
        exec bash
        ;;
    *)
        exec "$@"
        ;;
esac

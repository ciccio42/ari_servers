#!/bin/bash
# Entrypoint of the ari_whisper_server images. ari_servers is mounted as /workspace: the code of the server is
# read from /workspace/whisper at every start (a change needs only a restart of the container), the models from
# /workspace/models, the latency log goes to /workspace/latency_logs. No ROS, nothing to build.
#
# Usage: entrypoint.bash [server [whisper_server args] | shell | <command>]
#   server  (default) HTTP server: WHISPER_PORT (default 8000), SILERO_VAD_THRESHOLD (default 0.7), DEMO_LANGUAGE
set -e

export PYTHONPATH=$WORKSPACE/common:$WORKSPACE/whisper${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

case "${1:-server}" in
    server)
        shift || true
        cd "$WORKSPACE"
        exec python3 -m whisper_server "$@"
        ;;
    shell)
        exec bash
        ;;
    *)
        exec "$@"
        ;;
esac

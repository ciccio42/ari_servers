#!/bin/bash
# vLLM in Docker, the final configuration of the repository high-level-unisa (doc/resoconto_attivita_5.pdf,
# test_baseline/sistema.sh, avvia_vllm_esperimenti.sh, coda_rr_0511.sh) for the high_level_unisa RAG:
#   engine    (container vllm_server, section vllm):   Qwen3.6-35B-A3B NVFP4 served as qwen3.6-nvfp4 on vLLM 0.31, MTP with
#             3 proposed tokens, 16 requests together, 8192 tokens per step, 40% of the memory, prefix caching on.
#             Used by the LLM server (backend vllm) for the answers and by the RAG for its calls (rewriting, list mode)
#   reranker  (container vllm_reranker, section vllm_reranker): bge-reranker-v2-m3 on vLLM 0.31 (/rerank), fixed cache
#             of 0.5 GB (about 6.2 GB in all). Used by the RAG (RERANKER_BACKEND=vllm)
# Both are started and stopped by the orchestrator.
#
# Usage: bash run.bash <start|stop|status|logs|download> [engine|reranker]   (default: engine)
#   start     runs the container (the start of the engine takes minutes, the first time 15-20)
#   stop      stops and removes it (frees the memory)
#   status    container state and GET /v1/models
#   logs      follows the log
#   download  downloads the model in the Hugging Face cache of ari_servers (models/vllm, models/rag/.hf_cache), once
#
# Configuration: sections vllm and vllm_reranker of config/servers.yaml. Memory (sistema.sh): the engine reserves
# gpu_memory_utilization of the WHOLE memory, shared by CPU and GPU (0.40: about 49 GB); start refuses to run with less
# than gpu_memory_utilization x total + reserve_gb available (26 GB, as RISERVA_GB of sistema.sh).

COMMAND=$1
SERVICE=${2:-engine}
[ -n "$COMMAND" ] || { sed -n 13,19p "$0"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)   # ari_servers
CONFIG=$ROOT_DIR/${ARI_SERVERS_CONFIG:-config/servers.yaml}

# sections vllm / vllm_reranker of the configuration, defaults of sistema.sh and coda_rr_0511.sh
eval "$(python3 - "$CONFIG" "$SERVICE" <<'PY'
import json, shlex, sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
if sys.argv[2] == "reranker":
    s = c.get("vllm_reranker") or {}
    values = {"URL": s.get("url", "http://localhost:8002"),
              "IMAGE": s.get("image", "vllm/vllm-openai:v0.31.0"), "CACHE": s.get("cache_volume", "ari-vllm-cache-031"),
              "MODEL": s.get("model", "BAAI/bge-reranker-v2-m3"), "REVISION": s.get("revision", "main"),
              "HF_HOME_DIR": s.get("hf_home", "models/rag/.hf_cache"), "SERVED": s.get("served_model", "bge-reranker-v2-m3"),
              "MEM": s.get("gpu_memory_utilization", 0.05), "KV_BYTES": s.get("kv_cache_memory_bytes", 536870912),
              "MAX_LEN": s.get("max_model_len", 8192), "MAX_SEQS": s.get("max_num_seqs", 256),
              "MIN_FREE_GB": s.get("min_free_gb", 20)}
else:
    s = c.get("vllm") or {}
    values = {"URL": s.get("url", "http://localhost:8001"), "IMAGE": s.get("image", "vllm/vllm-openai:v0.31.0"),
              "CACHE": s.get("cache_volume", "ari-vllm-cache-031"), "MODEL": s.get("model", "nvidia/Qwen3.6-35B-A3B-NVFP4"),
              "REVISION": s.get("revision", "main"), "HF_HOME_DIR": s.get("hf_home", "models/vllm"),
              "SERVED": s.get("served_model", "qwen3.6-nvfp4"), "MEM": s.get("gpu_memory_utilization", 0.40),
              "MAX_LEN": s.get("max_model_len", 131072), "MAX_SEQS": s.get("max_num_seqs", 16),
              "MAX_BATCHED": s.get("max_num_batched_tokens", 8192),
              "SPEC": json.dumps(s["speculative_config"]) if s.get("speculative_config") else "",
              "PREFIX_CACHING": str(s.get("prefix_caching", True)).lower(), "RESERVE_GB": s.get("reserve_gb", 26)}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)" || { echo "$CONFIG: cannot read the sections vllm / vllm_reranker" >&2; exit 1; }

case "$SERVICE" in
    engine) CONTAINER=vllm_server ;;
    reranker) CONTAINER=vllm_reranker ;;
    *) echo "unknown service $SERVICE: engine or reranker" >&2; exit 1 ;;
esac
HOSTPORT=${URL#*://}
PORT=${HOSTPORT##*:}
HF_DIR=$ROOT_DIR/$HF_HOME_DIR   # HF_HOME: hub/models--<org>--<name>/snapshots/<revision>
MODEL_DIR=$HF_DIR/hub/models--${MODEL//\//--}

snapshot() {
    [ "$REVISION" != main ] && [ -d "$MODEL_DIR/snapshots/$REVISION" ] && { echo "$REVISION"; return; }
    ls "$MODEL_DIR/snapshots" 2> /dev/null | head -1
}
available_gb() { free -g | awk '/^Mem:/ {print $7}'; }

case "$COMMAND" in
    download)
        mkdir -p "$HF_DIR"
        exec docker run --rm -v "$ROOT_DIR":/workspace -e HF_HOME="/workspace/$HF_HOME_DIR" ari_rag_server bash -c \
            "conda run --no-capture-output -n rag_high_level_unisa hf download $MODEL --revision $REVISION \
             && chmod -R a+rwX /workspace/$HF_HOME_DIR"
        ;;
    start)
        SNAP=$(snapshot)
        [ -n "$SNAP" ] || { echo "model $MODEL not found in $MODEL_DIR: bash $0 download $SERVICE" >&2; exit 1; }
        if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
            echo "container $CONTAINER exists: bash $0 stop $SERVICE" >&2; exit 1
        fi
        AVAILABLE=$(available_gb)
        if [ "$SERVICE" = engine ]; then
            NEEDED=$(awk -v m="$MEM" -v t="$(free -g | awk '/^Mem:/ {print $2}')" -v r="$RESERVE_GB" 'BEGIN { printf "%d", m * t + r + 0.999 }')
        else
            NEEDED=$MIN_FREE_GB
        fi
        if [ "${AVAILABLE:-0}" -lt "$NEEDED" ]; then
            echo "only ${AVAILABLE} GB of memory available, $CONTAINER needs $NEEDED: unload the other models" >&2
            exit 1
        fi
        # network host on 127.0.0.1 as the other servers (there: -p 127.0.0.1:<port>:8000)
        if [ "$SERVICE" = engine ]; then
            ARGS=(--model "/modello/snapshots/$SNAP" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT"
                  --trust-remote-code --quantization modelopt --moe-backend marlin --attention-backend flashinfer
                  --kv-cache-dtype fp8 --gpu-memory-utilization "$MEM" --max-model-len "$MAX_LEN"
                  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_BATCHED")
            [ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
            [ "$PREFIX_CACHING" = false ] && ARGS+=(--no-enable-prefix-caching)
            docker run -d --name "$CONTAINER" --gpus all --shm-size 16g --network host \
                -v "$MODEL_DIR":/modello:ro -v "$CACHE":/root/.cache \
                -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_NVFP4_GEMM_BACKEND=marlin \
                "$IMAGE" "${ARGS[@]}" || exit 1
        else
            docker run -d --name "$CONTAINER" --gpus all --shm-size 4g --network host \
                -v "$MODEL_DIR":/modello:ro -v "$CACHE":/root/.cache \
                -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
                "$IMAGE" --model "/modello/snapshots/$SNAP" --served-model-name "$SERVED" \
                --host 127.0.0.1 --port "$PORT" --runner pooling \
                --gpu-memory-utilization "$MEM" --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
                --kv-cache-memory-bytes "$KV_BYTES" || exit 1
        fi
        echo "$CONTAINER starting (compile caches in the Docker volume $CACHE): bash $0 status $SERVICE"
        ;;
    stop)
        docker rm -f "$CONTAINER"
        ;;
    status)
        docker ps -a --filter "name=^${CONTAINER}$" --format '{{.Names}} {{.Image}} {{.Status}}'
        curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" && echo || { echo "no answer at port $PORT" >&2; exit 1; }
        ;;
    logs)
        exec docker logs -f "$CONTAINER"
        ;;
    *)
        sed -n 13,19p "$0"; exit 1
        ;;
esac

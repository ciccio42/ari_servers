# Orchestrator of the models (no ROS, no GPU): Python (standard library + PyYAML for servers.yaml), bash, curl and the
# Docker client, to start and stop the RAG containers and Ollama with their run.bash through the socket of Docker.
# Build from ari_servers: bash orchestrator/docker/run.bash build
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-yaml docker.io \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.bash /opt/orchestrator/entrypoint.bash
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/opt/orchestrator/entrypoint.bash"]

# Image of the RAG servers (no ROS): one conda environment per RAG (rag/<name>/environment.yml -> rag_<name>), so
# RAGs with different dependencies run from the same image, each in its own container. The code is not copied: it
# is read from /workspace (ari_servers mounted) at every start.
# Build from ari_servers: bash rag/docker/run.bash build (context: rag/, see rag/.dockerignore)
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda-forge only), for the architecture of the machine (aarch64 on the Spark)
ARG MINIFORGE_VERSION=25.3.1-0
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# one environment per RAG: a change of one environment.yml rebuilds from here
COPY . /opt/rag/
RUN for env in /opt/rag/*/environment.yml; do \
        echo "== $env" && conda env create -f "$env" || exit 1; \
    done && conda clean -afy && rm -rf /root/.cache/pip

ENV WORKSPACE=/workspace PYTHONUNBUFFERED=1
ENTRYPOINT ["/opt/rag/docker/entrypoint.bash"]

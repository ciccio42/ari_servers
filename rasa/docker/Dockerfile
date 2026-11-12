# RASA server: the three bots, the action server of the social bot and the proxy with the execution lock.
# CPU only: the bots are small (DIET, spaCy sm), the GPU stays to Whisper and the LLM.
# Build from ari_servers: bash rasa/docker/run.bash build
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Rome

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential libhdf5-dev pkg-config curl ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/rasa/requirements.txt
RUN python3 -m pip install -U "pip<25" setuptools wheel \
    && python3 -m pip install -r /opt/rasa/requirements.txt

COPY entrypoint.bash /opt/rasa/entrypoint.bash
ENV WORKSPACE=/workspace
ENTRYPOINT ["/opt/rasa/entrypoint.bash"]

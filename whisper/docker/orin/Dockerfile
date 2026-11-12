# Whisper HTTP server on the Jetson AGX Orin (L4T R36.5, JetPack 6, CUDA 12.6). No ROS.
# The base is the ONNX Runtime image built with jetson-containers.
# Build on the Orin from whisper/docker: bash run.bash orin build
ARG BASE_IMAGE=onnxruntime:r36.4.tegra-aarch64-cu126-22.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

# Older bases point to pypi.jetson-ai-lab.dev, removed from the web (410 Gone)
ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 \
    UV_DEFAULT_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126 \
    PIP_TRUSTED_HOST=jetson-ai-lab.io

# same versions of ari_demo (ari_whisper:orin), the ones of the tests of the report
RUN python3 -m pip install torch==2.11.0
RUN python3 -m pip install 'numpy<2,>=1.26' transformers==4.53.2 optimum==2.1.0 optimum-onnx soundfile scipy
# without its dependencies: they would bring the CPU onnxruntime and torchaudio
RUN python3 -m pip install --no-deps silero-vad
# torch 2.11 of jetson-ai-lab links libopenblas.so.0 and libcudss.so.0 (cuDSS), missing in the base and in JetPack:
# without them torch, transformers and silero fail. cuDSS 0.7.1 for Tegra, as jetson-containers (cudastack)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libsndfile1 libopenblas0 wget \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcudss0-cuda-12_0.7.1.4-1_arm64.deb -O /tmp/cudss.deb \
    && apt-get install -y --no-install-recommends /tmp/cudss.deb \
    && echo /usr/lib/aarch64-linux-gnu/libcudss/12 > /etc/ld.so.conf.d/cudss.conf \
    && ldconfig && rm -rf /tmp/cudss.deb /var/lib/apt/lists/*

COPY entrypoint.bash /opt/whisper/entrypoint.bash
ENV WORKSPACE=/workspace
ENTRYPOINT ["/opt/whisper/entrypoint.bash"]

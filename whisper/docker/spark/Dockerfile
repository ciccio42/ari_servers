# Whisper HTTP server on the DGX Spark (GB10, aarch64 sbsa, driver CUDA 13.0). No ROS.
# The base comes from jetson-containers: ONNX Runtime built with the GB10 kernels. The wheels for JetPack have only
# the kernels of the Orin GPU and fail with cudaErrorNoKernelImageForDevice. Built by run.bash (spark build), with
# jetson-containers-cu130.patch applied to jetson-containers.
# Build from whisper/docker: bash run.bash spark build
ARG BASE_IMAGE=ari_onnxruntime:spark-r39.1.arm64-sbsa-cu130-24.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# same versions of ari_demo (ari_whisper:spark), the ones of the tests of the report
RUN python3 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
RUN python3 -m pip install 'numpy<2,>=1.26' transformers==4.53.2 optimum==2.1.0 optimum-onnx soundfile scipy
# without its dependencies: they would bring the CPU onnxruntime and torchaudio
RUN python3 -m pip install --no-deps silero-vad

COPY entrypoint.bash /opt/whisper/entrypoint.bash
ENV WORKSPACE=/workspace
ENTRYPOINT ["/opt/whisper/entrypoint.bash"]

# Piper TTS server on the DGX Spark. The base is the image of the Whisper server: onnxruntime with CUDA is already
# there. piper-tts without its dependencies, so pip does not replace that onnxruntime with the CPU one.
# Build from ari_servers: bash tts/docker/run.bash build (after bash whisper/docker/run.bash spark build)
ARG BASE_IMAGE=ari_whisper_server:spark
FROM ${BASE_IMAGE}
RUN python3 -m pip install --no-deps piper-tts==1.8.0 pathvalidate
COPY entrypoint.bash /opt/tts/entrypoint.bash
ENV WORKSPACE=/workspace
ENTRYPOINT ["/opt/tts/entrypoint.bash"]

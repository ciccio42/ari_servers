# LLM (+RAG) server on the DGX Spark. The base is the image of the Whisper server: torch for CUDA 13 and
# transformers are already there, for the local backends (transformers, to port). The ChatGPT backend needs openai.
# Build from ari_servers: bash llm/docker/run.bash build (after bash whisper/docker/run.bash spark build)
ARG BASE_IMAGE=ari_whisper_server:spark
FROM ${BASE_IMAGE}
RUN python3 -m pip install openai==1.91.0
COPY entrypoint.bash /opt/llm/entrypoint.bash
ENV WORKSPACE=/workspace
ENTRYPOINT ["/opt/llm/entrypoint.bash"]

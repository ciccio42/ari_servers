# Whisper HTTP server on the internal Jetson of ARI (AGX Orin 32GB, JetPack 5.1.1, L4T R35.3.1, CUDA 11.4).
# JetPack 5 has Python 3.8 and no wheels for the versions of the Spark and the Orin: the base is the demo image
# pepper_demo:r35, already on the Jetson, with torch 2.0 (nv23.05), onnxruntime-gpu 1.16.1, transformers 4.46.3,
# optimum 1.23.3 and silero-vad. That image has ROS too: the server does not use it.
# Build on the Jetson from whisper/docker: bash run.bash jp5 build
ARG BASE_IMAGE=pepper_demo:r35
FROM ${BASE_IMAGE}

# for the tools (bench_http.py reads WAV files); the wheel brings its own libsndfile
RUN python3 -m pip install --no-cache-dir soundfile

COPY entrypoint.bash /opt/whisper/entrypoint.bash
ENV WORKSPACE=/workspace
ENTRYPOINT ["/opt/whisper/entrypoint.bash"]

"""Benchmark of a whisper_server over HTTP, without ROS: the protocol of audio_publisher, the requests of whisper_client.

The WAV files of a folder alternate at every round, as with audio_publisher. The audio goes out as whisper_client
sends it: mono float32 samples at 16 kHz (application/octet-stream), on one keep-alive connection with TCP_NODELAY.
The output has the columns of audio_publisher, so summarize_whisper.py joins it with the whisper_server.done lines
of the latency log (server on this machine). Here the client round trip is the HTTP round trip: no ROS overhead.
It also keeps the times of every reply of the server (server_total_s, server_decode_s, ...), for a server on
another machine.

Usage: bench_http.py [--url http://localhost:8000] [--audio-dir audio_files] [--runs 20] [--warmup 3] [--output <csv>]
Needs numpy and soundfile (environment ari_whisper).
"""
import argparse
import csv
import http.client
import json
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
# fields of the reply of the server, saved as server_<key>: with the server on another machine they need no clock sync
SERVER_TIMES = ("speech", "total_s", "decode_s", "queue_s", "vad_s", "asr_s")


def load(path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"{path}: expected mono audio at {SAMPLE_RATE} Hz")
    return audio


class Client:

    def __init__(self, url: str, timeout: float):
        parsed = urlparse(url)
        self.host, self.port, self.timeout = parsed.hostname, parsed.port or 80, timeout
        self.connection = None

    def post(self, audio: np.ndarray) -> dict:
        if self.connection is None:
            self.connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            self.connection.connect()
            self.connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.request("POST", "/transcribe", body=audio.tobytes(),
                                headers={"Content-Type": "application/octet-stream"})
        response = self.connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body.decode(errors='replace')}")
        return json.loads(body)


def main():
    workspace = Path(os.environ.get("WORKSPACE", "."))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("WHISPER_SERVER_URL", "http://localhost:8000"))
    parser.add_argument("--audio-dir", type=Path, default=workspace / "audio_files")
    parser.add_argument("--runs", type=int, default=20, help="repetitions of every file")
    parser.add_argument("--warmup", type=int, default=3, help="initial rounds marked to be left out of the statistics")
    parser.add_argument("--timeout", type=float, default=120, help="seconds for one request")
    parser.add_argument("--output", type=Path,
                        default=workspace / "latency_logs" / f"whisper_runs_{socket.gethostname()}_http.csv")
    args = parser.parse_args()

    wavs = sorted(args.audio_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No WAV files in {args.audio_dir}")
    audios = [load(wav) for wav in wavs]
    client = Client(args.url, args.timeout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "round", "warmup", "samples", "epoch_start_ns", "epoch_end_ns", "round_trip_s",
                         *(f"server_{key}" for key in SERVER_TIMES), "text"])
        for i in range(args.warmup + args.runs):
            for wav, audio in zip(wavs, audios):
                epoch_start = time.time_ns()
                start = time.perf_counter()
                reply = client.post(audio)
                elapsed = time.perf_counter() - start
                writer.writerow([wav.name, i, int(i < args.warmup), audio.size,
                                 epoch_start, time.time_ns(), round(elapsed, 4),
                                 *(reply.get(key, "") for key in SERVER_TIMES), reply["text"]])
                f.flush()
                print(f"[{i + 1}/{args.warmup + args.runs}] {wav.name} {elapsed:.3f} s "
                      f"(server {reply['total_s']:.3f} s): {reply['text']}", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

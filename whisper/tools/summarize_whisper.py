#!/usr/bin/env python3
"""Final metrics of a Whisper test: audio, times of the node and of the client, real time factor, transcripts.

Joins whisper_runs_<host>.csv (audio_publisher) with the `whisper.done` lines of latency_<date>_<host>.csv
through the time of the event (asr_node, or whisper_server and whisper_client for the network). With the server
on another machine the log of the client is enough: it keeps the times of the server reply, no clock sync needed. Variances are sample variances (n-1), percentiles are inclusive.
Warmup rounds are left out of the statistics and reported apart. Needs only the standard library.

Usage: summarize_whisper.py <audio_dir> <whisper_runs.csv> [<latency.csv> ...] [--csv <summary.csv>]
  no latency.csv: whisper_runs.csv of bench_http.py, with the times of the server replies (server_* columns)
  --csv  also writes one row per metric and file (all files included), to compare boards side by side
"""
import argparse
import csv
import socket
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path

ALL = "all files"
STATS = ["n", "mean", "variance", "std", "min", "p50", "p95", "max"]


def durations(audio_dir):
    result = {}
    for path in sorted(Path(audio_dir).glob("*.wav")):
        with wave.open(str(path)) as f:
            result[path.name] = f.getnframes() / f.getframerate()
    return result


MODEL_STAGES = ("whisper", "whisper_server")  # asr_node (local) or whisper_server (network)


def whisper_events(paths):
    """(epoch_ns, stage, extra) of the done events: model (asr_node or whisper_server) and whisper_client."""
    events = []
    for path in paths:
        with open(path, newline="") as f:
            for line in csv.reader(f):
                if len(line) > 6 and line[5] == "done" and line[4] in MODEL_STAGES + ("whisper_client",):
                    events.append((int(line[0]), line[4], dict(item.split("=", 1) for item in line[6].split(";"))))
    return events


def stats(values):
    n = len(values)
    variance = statistics.variance(values) if n > 1 else 0.0
    if n > 1:
        cuts = statistics.quantiles(values, n=100, method="inclusive")
        p50, p95 = cuts[49], cuts[94]
    else:
        p50 = p95 = values[0]
    return {"n": n, "mean": statistics.mean(values), "variance": variance, "std": variance ** 0.5,
            "min": min(values), "p50": p50, "p95": p95, "max": max(values)}


def row(name, s):
    return (f"{name:<36} {s['n']:>4} {s['mean']:>9.4f} {s['variance']:>11.6f} {s['std']:>9.4f} "
            f"{s['min']:>8.4f} {s['p50']:>8.4f} {s['p95']:>8.4f} {s['max']:>8.4f}")


HEADER = (f"{'':<36} {'n':>4} {'mean':>9} {'variance':>11} {'std':>9} "
          f"{'min':>8} {'p50':>8} {'p95':>8} {'max':>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_dir")
    parser.add_argument("runs_csv")
    parser.add_argument("latency_csv", nargs="*",
                        help="latency logs; none for a runs CSV of bench_http.py with the server_* columns")
    parser.add_argument("--csv", type=Path, help="summary file, one row per metric and file")
    args = parser.parse_args()

    audio = durations(args.audio_dir)
    events = whisper_events(args.latency_csv)

    # metric -> file -> values, steady rounds only
    metrics = defaultdict(lambda: defaultdict(list))
    warmup = defaultdict(list)
    texts = defaultdict(Counter)
    no_speech = 0
    with open(args.runs_csv, newline="") as f:
        for run in csv.DictReader(f):
            name = run["file"]
            if run["warmup"] == "1":
                warmup[name].append(float(run["round_trip_s"]))
                continue
            start, end = int(run["epoch_start_ns"]), int(run["epoch_end_ns"])
            matches = [extra for epoch, stage, extra in events if start <= epoch <= end and stage in MODEL_STAGES]
            client = [extra for epoch, stage, extra in events if start <= epoch <= end and stage == "whisper_client"]
            if not matches and len(client) == 1 and "server_vad_s" in client[0]:
                # server on another machine (its log not given, or its clock not in sync): the times of the reply
                c = client[0]
                matches = [{"speech": c["speech"], "asr_s": c["server_asr_s"], "vad_s": c["server_vad_s"],
                            "total_s": c["server_total_s"]}]
            if not matches and not client and run.get("server_total_s"):
                # bench_http.py against a server on another machine: the times of the reply are in the runs CSV
                matches = [{"speech": run["server_speech"], "asr_s": run["server_asr_s"], "vad_s": run["server_vad_s"],
                            "total_s": run["server_total_s"], "decode_s": run["server_decode_s"]}]
            if len(matches) != 1:
                raise SystemExit(f"{len(matches)} whisper(_server).done events for {name} round {run['round']}")
            node = matches[0]
            no_speech += node["speech"] != "True"
            texts[name][run["text"].strip()] += 1
            asr, vad, total = float(node["asr_s"]), float(node["vad_s"]), float(node["total_s"])
            round_trip = float(run["round_trip_s"])
            metrics["Inference, asr_s (s)"][name].append(asr)
            metrics["VAD, vad_s (s)"][name].append(vad)
            metrics["Node callback, total_s (s)"][name].append(total)
            metrics["Client round trip (s)"][name].append(round_trip)
            metrics["Overhead out of the model, round trip - total_s (s)"][name].append(round_trip - total)
            if "decode_s" in node:  # reception of the audio on the server, part of total_s
                metrics["Reception of the audio on the server, decode_s (s)"][name].append(float(node["decode_s"]))
            if len(client) == 1:  # network: whisper_client -> whisper_server
                http_s = float(client[0]["http_s"])
                metrics["HTTP round trip, whisper_client (s)"][name].append(http_s)
                metrics["Network, HTTP round trip - server total_s (s)"][name].append(http_s - total)
                metrics["ROS overhead, round trip - HTTP round trip (s)"][name].append(round_trip - http_s)
            metrics["Real time factor, asr_s / audio"][name].append(asr / audio[name])
            metrics["Real time factor, round trip / audio"][name].append(round_trip / audio[name])

    if not metrics:
        raise SystemExit("no steady rounds in " + args.runs_csv)

    summary = []  # (metric, file, stats)
    print("Audio duration (s)\n" + HEADER)
    print(row(ALL, stats(list(audio.values()))))
    for name, value in audio.items():
        print(f"  {name}: {value:.3f} s")

    for metric, per_file in metrics.items():
        print(f"\n{metric}\n{HEADER}")
        for name, values in per_file.items():
            s = stats(values)
            summary.append((metric, name, s))
            print(row(name, s))
        s = stats([v for values in per_file.values() for v in values])
        summary.append((metric, ALL, s))
        print(row(ALL, s))

    if warmup:
        print(f"\nWarmup rounds, client round trip (s): left out above\n{HEADER}")
        for name, values in warmup.items():
            print(row(name, stats(values)))
        first = [values[0] for values in warmup.values()]
        print(f"  first call of the test: {first[0]:.4f} s, first call of every file: mean {statistics.mean(first):.4f} s")

    print("\nTranscripts (steady rounds)")
    unstable = 0
    for name, counter in texts.items():
        text, count = counter.most_common(1)[0]
        total = sum(counter.values())
        unstable += len(counter) > 1
        print(f"  {name}: {count}/{total} identical{'' if len(counter) == 1 else f', {len(counter)} variants'}: {text}")

    steady = len(next(iter(metrics.values()))[next(iter(audio))]) if audio else 0
    print(f"\nSteady rounds per file: {steady}, calls with no speech detected: {no_speech}, "
          f"files with different transcripts: {unstable}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["host", "metric", "file"] + STATS)
            stem = Path(args.runs_csv).stem  # whisper_runs_<host>: the board of the test, not this machine
            host = stem.removeprefix("whisper_runs_") if stem.startswith("whisper_runs_") else socket.gethostname()
            for metric, name, s in summary:
                writer.writerow([host, metric, name] + [s["n"]] + [f"{s[k]:.6f}" for k in STATS[1:]])
        print(f"Summary written to {args.csv}")


if __name__ == "__main__":
    main()

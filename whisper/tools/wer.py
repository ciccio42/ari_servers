#!/usr/bin/env python3
"""Word and character error rate of the transcripts of bench_http.py, against audio_files/references.csv.

Normalization as the basic one of Whisper for non-English text: lowercase, apostrophes and punctuation become
spaces, accents and digits stay ("dell'università" -> "dell università", "GLP-1" -> "glp 1"). WER and CER are
corpus rates: edits (substitutions + deletions + insertions) of all the measured calls over the words (characters)
of their references. Warmup rounds are left out. Standard library only.

Usage: wer.py [--references audio_files/references.csv] <runs.csv> [<runs.csv> ...] [--per-file] [--csv <out.csv>]
"""
import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]|_", " ", text)  # \w keeps accented letters and digits
    return " ".join(text.split())


def edits(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the minimum Levenshtein alignment."""
    # each cell: (cost, s, d, i)
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            cur.append(min((sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                           (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                           (ins[0] + 1, ins[1], ins[2], ins[3] + 1)))
        prev = cur
    return prev[-1][1:]


def score(pairs):
    """pairs: [(reference, hypothesis)] normalized. WER, CER and S/D/I counts over all of them."""
    words = chars = 0
    s = d = i = char_edits = exact = 0
    for ref, hyp in pairs:
        ws, wd, wi = edits(ref.split(), hyp.split())
        s, d, i = s + ws, d + wd, i + wi
        words += len(ref.split())
        char_edits += sum(edits(list(ref), list(hyp)))
        chars += len(ref)
        exact += ref == hyp
    return {"n": len(pairs), "wer": (s + d + i) / words, "cer": char_edits / chars, "sub": s, "del": d, "ins": i,
            "exact": exact / len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--references", type=Path, default=Path("audio_files/references.csv"))
    parser.add_argument("--per-file", action="store_true", help="also the WER of every file")
    parser.add_argument("--csv", type=Path, help="one row per runs file (and per audio file with --per-file)")
    args = parser.parse_args()

    with open(args.references, newline="") as f:
        references = {row["file"]: normalize(row["text"]) for row in csv.DictReader(f)}

    rows = []
    print(f"{'runs':<28} {'n':>4} {'WER':>7} {'CER':>7} {'S':>4} {'D':>4} {'I':>4} {'exact':>6}")
    for path in args.runs:
        by_file = defaultdict(list)
        with open(path, newline="") as f:
            for run in csv.DictReader(f):
                if run["warmup"] == "1":
                    continue
                if run["file"] not in references:
                    raise SystemExit(f"{run['file']}: no reference in {args.references}")
                by_file[run["file"]].append((references[run["file"]], normalize(run["text"])))
        name = path.stem.removeprefix("runs_")
        total = score([pair for pairs in by_file.values() for pair in pairs])
        rows.append({"runs": name, "file": "all files", **total})
        print(f"{name:<28} {total['n']:>4} {total['wer']:>7.1%} {total['cer']:>7.1%} {total['sub']:>4} "
              f"{total['del']:>4} {total['ins']:>4} {total['exact']:>6.0%}")
        if args.per_file:
            for file, pairs in sorted(by_file.items()):
                result = score(pairs)
                rows.append({"runs": name, "file": file, **result})
                print(f"  {file:<34} WER {result['wer']:>6.1%} CER {result['cer']:>6.1%}  {pairs[0][1]}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written {args.csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Human review of SoussVoice V4 candidates. Stdlib only; no models or networking.

Original sampled and source-audit files are read-only. Decisions are saved after
EVERY reviewed entry, atomically, into a separate native_decisions/ directory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import tempfile
from pathlib import Path

SOURCES = {
    "latin": ("awal_latin_review_sample.csv", ("english", "tachelhit")),
    "tifinagh": ("awal_tifinagh_review_sample.csv", ("english", "tachelhit")),
    "darija": ("darija_review_sample.csv", ("tachelhit_arabic", "darija_arabic")),
}
DECISIONS = {"a": "accurate", "w": "wrong", "u": "uncertain", "s": "skip"}
EXTRA = ["native_decision", "corrected_tachelhit", "review_notes"]


def read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_atomic(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".review_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def review_one(root: Path, source: str):
    filename, display = SOURCES[source]
    original = root / filename
    if not original.is_file():
        raise FileNotFoundError(f"Missing: {original}")
    rows, fields = read(original)
    if not rows or not all(k in fields for k in ("source_id", *display)):
        raise ValueError(f"Unexpected or empty CSV schema: {original}: {fields}")
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Source IDs in the sampled input are not unique")
    fieldnames = list(dict.fromkeys(fields + EXTRA))
    destination = root / "native_decisions" / filename
    saved = {}
    if destination.exists():
        existing, _ = read(destination)
        saved = {r["source_id"]: r for r in existing}
        if set(saved) != {r["source_id"] for r in rows}:
            raise ValueError("Review sample IDs changed since review began; stop to reconcile")
        for r in rows:
            old = saved[r["source_id"]]
            if any(old.get(k) != r.get(k) for k in display):
                raise ValueError("Review sample changed since review began; stop to reconcile")
    result = []
    for row in rows:
        record = dict(row)
        old = saved.get(row["source_id"], {})
        for field in EXTRA:
            record[field] = old.get(field, record.get(field, ""))
        result.append(record)
    counters = collections.Counter(r["native_decision"] for r in result)
    print(f"\n=== {source.upper()} NATIVE REVIEW ===")
    print(f"Candidates: {len(result)}; previously rated: {len(result) - counters['']}")
    print("[a] Accurate  [w] Wrong  [u] Unsure  [s] Skip  [q] Quit")
    print("For inaccurate items, you can optionally enter a correction or note.")
    for idx, row in enumerate(result, 1):
        if row["native_decision"]:
            continue
        print(f"\n--- {source.upper()} {idx}/{len(result)} ---")
        print(f"Source: {row.get('source_id', '')}")
        if source == "darija":
            print("Tachelhit (Arabic):", row[display[0]])
            print("Darija:           ", row[display[1]])
            if row.get("review_note"):
                print("NOTE:", row["review_note"])
        else:
            print("English: ", row[display[0]])
            print("Tachelhit:", row[display[1]])
        while True:
            try:
                answer = input("Rating [a/w/u/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nReview interrupted; prior ratings are saved.")
                return False
            if answer in (*DECISIONS, "q"):
                break
            print("Choose a, w, u, s, or q.")
        if answer == "q":
            print("Exiting; prior ratings remain saved.")
            return False
        row["native_decision"] = DECISIONS[answer]
        if answer == "w":
            try:
                if source != "darija":
                    row["corrected_tachelhit"] = input("Better Tachelhit (optional): ").strip()
                row["review_notes"] = input("What is wrong? (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nNote entry interrupted; saving rating.")
        write_atomic(destination, result, fieldnames)
        print(f"Saved: {row['native_decision']}")
    totals = collections.Counter(r["native_decision"] for r in result)
    print(f"\n{source.upper()} REVIEW COMPLETE: " + ", ".join(f"{k}={totals[k]}" for k in ("accurate", "wrong", "uncertain", "skip")))
    print(f"Saved separate decision file: {destination}")
    return True


def self_test():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "review.csv"
        input_rows = [{"source_id": "abc", "english": "I want to eat", "tachelhit": "Riɣ ad ccɣ", "native_decision": "accurate", "review_notes": ""}]
        fields = list(input_rows[0])
        write_atomic(p, input_rows, fields)
        got, got_fields = read(p)
        assert got == input_rows and got_fields == fields
        assert DECISIONS["w"] == "wrong"
        assert len(SOURCES) == 3
    print("SELF-TEST PASS: safe CSV round-trip, atomic write, decision labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="latin")
    parser.add_argument("--review-dir", type=Path, default=Path("datasets/understanding_v4/native_review_batch_01"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.review_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Review directory not found: {root}")
    which = list(SOURCES) if args.source == "all" else [args.source]
    for source in which:
        if not review_one(root, source):
            break
    print("\nNo model training, no original CSV modifications, no test inference.")


if __name__ == "__main__":
    main()

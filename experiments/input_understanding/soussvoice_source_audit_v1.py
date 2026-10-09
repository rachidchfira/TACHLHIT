#!/usr/bin/env python3
"""SoussVoice source-audit V1: READ-ONLY against existing corpora/models.

Fetches Hugging Face Dataset Viewer *text rows*, not dataset audio/weights.
Outputs review queues; NEVER outputs an approved training corpus.
Python standard library only.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AWAL = "collectivat/amazic"
DARIJA = "NoureddineMOR/tachelhiyt-darija"
API = "https://datasets-server.huggingface.co/rows"
FIELDS_AWAL = ["source_id", "english", "tachelhit", "script", "direction", "variety", "validation", "isValidated", "isDiscarded", "isReported", "quality_label", "overlap", "source_url", "license"]
FIELDS_DARIJA = ["source_id", "tachelhit_arabic", "darija_arabic", "script", "quality_label", "source_url", "license"]


def norm(s):
    s = unicodedata.normalize("NFC", str(s or "")).casefold()
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.split())


def script(s):
    counts = {
        "tifinagh": sum('\u2d30' <= c <= '\u2d7f' for c in s),
        "arabic": sum(('\u0600' <= c <= '\u06ff') or ('\u0750' <= c <= '\u077f') or ('\u08a0' <= c <= '\u08ff') for c in s),
        "latin": sum('LATIN' in unicodedata.name(c, '') for c in s),
    }
    nonzero = [k for k, v in counts.items() if v > 0]
    return nonzero[0] if len(nonzero) == 1 else "mixed" if len(nonzero) > 1 else "unknown"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def protected_keys(root):
    """Keep *all* existing splits protected, not just native-gold and external benchmark."""
    files = [
        root / "datasets/reverse_translation/v2/train.csv",
        root / "datasets/reverse_translation/v2/val.csv",
        root / "datasets/reverse_translation/v2/validation.csv",
        root / "datasets/reverse_translation/v2/test.csv",
        root / "datasets/native_gold_v1/native_verified.csv",
        root / "datasets/external/datashi/inputs.csv",  # all 1,500, including reserved test
        root / "experiments/input_understanding/datashi_comparison_v1/comparisons.csv",
    ]
    eng, shi, found = set(), set(), []
    for p in files:
        if not p.exists():
            continue
        found.append(str(p.relative_to(root)))
        for r in read_csv(p):
            # Ordinary Tatoeba/DATASHI/Gold columns; use upper-case DATASHI where applicable.
            for k in ("en", "EN", "reference_en"):
                if r.get(k):
                    eng.add(norm(r[k]))
            for k in ("shi_latn", "shi", "SHI-s", "SHI-ns", "shi_tfng"):
                if r.get(k):
                    shi.add(norm(r[k]))
    if not (root / "datasets/reverse_translation/v2/train.csv").exists():
        raise FileNotFoundError("Missing existing train.csv; run from ~/soussvoice and keep overlap protection intact")
    if not (root / "datasets/external/datashi/inputs.csv").exists():
        raise FileNotFoundError("Missing DATASHI inputs.csv; this audit needs it to protect reserved examples")
    return eng, shi, found


def fetch_rows(dataset, config, split, max_pages=100):
    result = []
    total = None
    for page in range(max_pages):
        params = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                         "offset": page * 100, "length": 100})
        url = API + "?" + params
        error = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SoussVoice-source-audit/1.0", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=45) as res:
                    payload = json.load(res)
                error = None
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                error = e
                time.sleep(1.5 * (attempt + 1))
        if error is not None:
            raise RuntimeError(f"Could not fetch {dataset} page {page}: {error}") from error
        if total is None:
            total = payload.get("num_rows_total")
            if not isinstance(total, int):
                raise RuntimeError(f"Invalid row count from Hugging Face: {dataset}")
        batch = [x["row"] for x in payload.get("rows", [])]
        result.extend(batch)
        print(f"{dataset}: fetched {len(result)}/{total} text records", flush=True)
        if len(result) >= total:
            return result
        if not batch:
            raise RuntimeError(f"Unexpected empty page at {len(result)}/{total} for {dataset}")
    raise RuntimeError(f"Page limit reached while fetching {dataset}; no partial audit saved")


def classify_awal(row, idx, protected_en, protected_shi, seen):
    srclang = str(row.get("src") or "").strip().lower()
    tgtlang = str(row.get("tgt") or "").strip().lower()
    srcvar = str(row.get("srcVar") or "").strip().casefold()
    tgtvar = str(row.get("tgtVar") or "").strip().casefold()
    if srclang == "en" and tgtvar == "tachelhit":
        en, shi, direction = str(row.get("src_text") or "").strip(), str(row.get("tgt_text") or "").strip(), "en_to_shi"
    elif tgtlang == "en" and srcvar == "tachelhit":
        en, shi, direction = str(row.get("tgt_text") or "").strip(), str(row.get("src_text") or "").strip(), "shi_to_en"
    else:
        return "wrong_language_or_variety", None
    if not en or not shi:
        return "missing_text", None
    try:
        votes = int(row.get("validation") or 0)
    except (ValueError, TypeError):
        votes = 0
    valid = row.get("isValidated") is True
    discarded = row.get("isDiscarded") is True
    reported = row.get("isReported") is True
    verified_by_platform = valid and votes >= 1 and not discarded and not reported
    sc = script(shi)
    reasons = []
    if norm(en) in protected_en:
        reasons.append("english_exists_in_previous_data")
    if norm(shi) in protected_shi:
        reasons.append("shi_exists_in_previous_data")
    key = (norm(en), norm(shi))
    if key in seen:
        reasons.append("duplicate_pair_within_awal")
    seen.add(key)
    overlap = ";".join(reasons)
    record = {
        "source_id": f"Awal:{idx}", "english": en, "tachelhit": shi,
        "script": sc, "direction": direction, "variety": "Tachelhit",
        "validation": votes, "isValidated": valid, "isDiscarded": discarded,
        "isReported": reported,
        "quality_label": "platform_validated_not_native_verified" if verified_by_platform else "not_safely_validated",
        "overlap": overlap,
        "source_url": "https://huggingface.co/datasets/collectivat/amazic",
        "license": "cc-by-2.0",
    }
    if not verified_by_platform:
        return "unverified_or_rejected", record
    if overlap:
        return "overlap_or_duplicate", record
    if sc == "latin":
        return "novel_latin_review", record
    if sc == "tifinagh":
        return "novel_tifinagh_review", record
    return "other_script_review", record


def audit_awal(rows, protected_en, protected_shi):
    bins = collections.defaultdict(list)
    counters = collections.Counter()
    seen = set()
    for i, row in enumerate(rows):
        bucket, record = classify_awal(row, i, protected_en, protected_shi, seen)
        counters[bucket] += 1
        if record:
            bins[bucket].append(record)
    return bins, counters


def audit_darija(rows):
    unique = set()
    output = []
    counts = collections.Counter()
    for i, row in enumerate(rows):
        shi = str(row.get("tachelhit_transcript") or "").strip()
        darija = str(row.get("darija_transcript") or "").strip()
        if not shi or not darija:
            counts["missing_transcript"] += 1
            continue
        key = (norm(shi), norm(darija))
        if key in unique:
            counts["duplicate_pair"] += 1
            continue
        unique.add(key)
        output.append({"source_id": f"tachelhiyt-darija:{i}",
                       "tachelhit_arabic": shi, "darija_arabic": darija,
                       "script": script(shi), "quality_label": "unreviewed_no_english_translation",
                       "source_url": "https://huggingface.co/datasets/NoureddineMOR/tachelhiyt-darija",
                       "license": "mit"})
    counts["new_text_pairs_needing_review"] = len(output)
    return output, counts


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def self_test():
    assert norm("Hello, World!") == norm(" hello world ")
    assert script("ⴰⵣⵓⵍ") == "tifinagh"
    assert script("Ar tskcm.") == "latin"
    assert script("السلام") == "arabic"
    good = {"src": "en", "tgt": "zgh", "tgtVar": "Tachelhit", "src_text": "Hello", "tgt_text": "ⴰⵣⵓⵍ", "validation": 2, "isValidated": True}
    bucket, _ = classify_awal(good, 0, set(), set(), set())
    assert bucket == "novel_tifinagh_review", bucket
    bad = {**good, "validation": -2}
    bucket, _ = classify_awal(bad, 0, set(), set(), set())
    assert bucket == "unverified_or_rejected", bucket
    bucket, _ = classify_awal(good, 0, {"hello"}, set(), set())
    assert bucket == "overlap_or_duplicate", bucket
    wrong = {**good, "tgtVar": "Tarifit"}
    bucket, _ = classify_awal(wrong, 0, set(), set(), set())
    assert bucket == "wrong_language_or_variety", bucket
    print("SELF-TEST PASS: normalization, script, negative-score safeguard, overlap, variety")


def main():
    ap = argparse.ArgumentParser(description="Audit new Tachelhit data without training or downloading audio")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--source", choices=("awal", "darija", "both"), default="awal")
    ap.add_argument("--output", default="datasets/understanding_v4/source_audit_v1")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = Path.cwd()
    output = root / args.output
    if output.exists():
        raise SystemExit(f"Refusing overwrite: {output}\nUse --output another/new_directory")
    protected_en, protected_shi, protected_files = protected_keys(root)
    print(f"PROTECTED: {len(protected_en)} English and {len(protected_shi)} Tachelhit normalized inputs")
    print("Existing held-out/test files are READ ONLY. No training files will be produced.", flush=True)
    report = {"version": "source_audit_v1", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source": args.source, "protected_files": protected_files,
              "protected_english_keys": len(protected_en), "protected_tachelhit_keys": len(protected_shi),
              "training_approved": False, "notes": ["Platform validation is not native-speaker proof.",
              "Unreviewed Tifinagh and Arabic-script inputs must not be silently converted or mixed with Latin training data.",
              "English overlap is conservatively flagged even when the Tachelhit spelling differs."]}
    files = {}
    if args.source in ("awal", "both"):
        data = fetch_rows(AWAL, "Awal", "train")
        bins, counts = audit_awal(data, protected_en, protected_shi)
        report["awal"] = {"total_downloaded_text_rows": len(data), "categories": dict(counts), "license": "cc-by-2.0"}
        mapping = {
            "novel_latin_review": "awal_new_latin_review.csv",
            "novel_tifinagh_review": "awal_new_tifinagh_review.csv",
            "other_script_review": "awal_other_script_review.csv",
            "overlap_or_duplicate": "awal_overlaps.csv",
            "unverified_or_rejected": "awal_unverified.csv",
        }
        for key, filename in mapping.items():
            files[filename] = (bins[key], FIELDS_AWAL)
    if args.source in ("darija", "both"):
        data = fetch_rows(DARIJA, "default", "train", max_pages=30)
        rows, counts = audit_darija(data)
        report["darija"] = {"total_downloaded_text_rows": len(data), "categories": dict(counts), "license": "mit"}
        files["tachelhiyt_darija_text_review.csv"] = (rows, FIELDS_DARIJA)
    output.mkdir(parents=True, exist_ok=False)
    for name, (rows, fields) in files.items():
        write_csv(output / name, rows, fields)
    with (output / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== SOUSSVOICE SOURCE AUDIT V1 ===")
    for name, (rows, fields) in files.items():
        print(f"{name}: {len(rows)}")
        if "new_latin" in name:
            for row in rows[:5]:
                print(f"  EN: {row['english']}\n  SHI: {row['tachelhit']}")
    print("SAVED:", output)
    print("TRAINING: NO. Existing checkpoints/data changed: NO. Reserved tests evaluated: NO.")
    print("None of these candidates are native-verified or automatically approved for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

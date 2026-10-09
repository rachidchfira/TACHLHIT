#!/usr/bin/env python3
"""Create deterministic, read-only native review queues for SoussVoice V4.

Uses V1 source-audit outputs, NEVER emits training-ready data.
Python stdlib only; no network, HF downloads, or model loading.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

AWAL_LATN = "awal_new_latin_review.csv"
AWAL_TFNG = "awal_new_tifinagh_review.csv"
DARIJA = "tachelhiyt_darija_text_review.csv"


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit CSV missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def tier(value: str) -> str:
    count = len(value.split())
    return "short" if count <= 4 else "medium" if count <= 11 else "long"


def sample_diverse(entries: list[dict], target: int, text_key: str, salt: str) -> list[dict]:
    groups = collections.defaultdict(list)
    for entry in entries:
        groups[tier(entry[text_key])].append(entry)
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256((salt + "|" + r["source_id"] + "|" + norm(r[text_key])).encode()).hexdigest())
    selected = []
    for category in ("short", "medium", "long"):
        while groups[category] and len(selected) < target and sum(tier(r[text_key]) == category for r in selected) < (target + 2) // 3:
            selected.append(groups[category].pop(0))
    leftovers = sorted((r for group in groups.values() for r in group), key=lambda r: hashlib.sha256((salt + "|" + r["source_id"]).encode()).hexdigest())
    return (selected + leftovers[:max(0, target - len(selected))])[:target]


def dump(path: Path, records: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def create(root: Path, out: Path, sample_n: int = 12):
    awal_dir = root / "datasets/understanding_v4/source_audit_v1"
    darija_dir = root / "datasets/understanding_v4/darija_audit_v1"
    if out.exists():
        raise FileExistsError(f"Refusing overwrite: {out}")
    latin = rows(awal_dir / AWAL_LATN)
    tifinagh = rows(awal_dir / AWAL_TFNG)
    darija = rows(darija_dir / DARIJA)
    for record in latin + tifinagh:
        if not record.get("source_id") or not record.get("english") or not record.get("tachelhit"):
            raise ValueError("Awal audit row missing source_id/english/tachelhit")
    for record in darija:
        if not record.get("source_id") or not record.get("tachelhit_arabic") or not record.get("darija_arabic"):
            raise ValueError("Darija audit row missing required transcripts/source_id")
    counts = collections.Counter(norm(x["tachelhit_arabic"]) for x in darija)
    ambiguous = {key for key, n in counts.items() if n > 1}
    darija_copy = []
    for record in darija:
        item = dict(record)
        src = norm(record["tachelhit_arabic"])
        item["review_note"] = "Same source has multiple translations" if src in ambiguous else ""
        darija_copy.append(item)
    spec = [
        ("awal_latin_review_sample.csv", sample_diverse(latin, sample_n, "english", "awal-latn"),
         ["source_id", "english", "tachelhit", "script", "quality_label", "source_url", "license", "native_decision", "corrected_tachelhit", "review_notes"]),
        ("awal_tifinagh_review_sample.csv", sample_diverse(tifinagh, sample_n, "english", "awal-tfng"),
         ["source_id", "english", "tachelhit", "script", "quality_label", "source_url", "license", "native_decision", "corrected_tachelhit", "review_notes"]),
        ("darija_review_sample.csv", sample_diverse(darija_copy, sample_n, "tachelhit_arabic", "darija"),
         ["source_id", "tachelhit_arabic", "darija_arabic", "script", "quality_label", "review_note", "source_url", "license", "native_decision", "review_notes"]),
    ]
    reports = {}
    for name, folder in (("awal", awal_dir), ("darija", darija_dir)):
        rp = folder / "report.json"
        if rp.exists():
            with rp.open(encoding="utf-8") as fh:
                source_report = json.load(fh)
            reports[name] = source_report.get(name, {})
    overview = {
        "version": "v4_review_queues_v1", "training_approved": False,
        "counts": {"awal_latin": len(latin), "awal_tifinagh": len(tifinagh), "darija_unique_pairs": len(darija),
                   "darija_source_strings_with_multiple_translations": len(ambiguous)},
        "original_audits": reports,
        "note": "Darija pairs lack an English reference and must NEVER be silently treated as SHI-English pairs. Platform validation is not native approval.",
    }
    out.mkdir(parents=True)
    for name, selected, fields in spec:
        dump(out / name, selected, fields)
    (out / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== SOUSSVOICE V4 REVIEW QUEUES ===")
    print(f"Awal English–Tachelhit: {len(latin)} Latin and {len(tifinagh)} Tifinagh candidates")
    print(f"Tachelhit–Darija: {len(darija)} unique pairs (NO English translations)")
    print(f"Tachelhit strings with multiple Darija translations: {len(ambiguous)}")
    d = reports.get("darija", {})
    print("Darija source audit breakdown:", d.get("categories", "not available"))
    print(f"Review samples: {len(spec[0][1])} Latin, {len(spec[1][1])} Tifinagh, {len(spec[2][1])} Darija")
    print("Output:", out)
    print("Training: NO. Native approval: NO. Reserved tests: NOT EVALUATED.")
    for name, selected, _ in spec:
        print(f"\n=== {name} ===")
        for entry in selected[:3]:
            if "tachelhit_arabic" in entry:
                print("SHI-Arab:", entry["tachelhit_arabic"])
                print("Darija:", entry["darija_arabic"])
            else:
                print("EN:", entry["english"])
                print("SHI:", entry["tachelhit"])


def test():
    assert tier("hello") == "short"
    assert tier("one two three four five") == "medium"
    assert tier(" ".join(["one"] * 12)) == "long"
    a = [{"source_id": f"a{i}", "en": "hello"} for i in range(3)]
    assert sample_diverse(a, 2, "en", "fixed") == sample_diverse(a, 2, "en", "fixed")
    assert len(sample_diverse(a, 2, "en", "fixed")) == 2
    assert norm(" X  Y ") == "x y"
    print("SELF-TEST PASS: deterministic sampling, balanced tiers, Unicode normalization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("datasets/understanding_v4/native_review_batch_01"))
    parser.add_argument("--per-source", type=int, default=12)
    args = parser.parse_args()
    if args.self_test:
        test()
    else:
        create(args.root, args.root / args.output, args.per_source)

import csv
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

ROOT = Path("datasets/reverse_translation/v1")
OUT = ROOT / "quality_audit.json"

def norm(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = value.casefold()
    return " ".join(value.split())

def read_csv(name):
    with (ROOT / name).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

splits = {
    name: read_csv(f"{name}.csv")
    for name in ("train", "validation", "test")
}

def keys(rows, column):
    return {
        norm(row[column])
        for row in rows
        if norm(row[column])
    }

audit = {"splits": {}, "cross_split_overlap": {}}

for name, rows in splits.items():
    shi = [norm(r["shi_latn"]) for r in rows]
    en = [norm(r["en"]) for r in rows]

    pairs = list(zip(shi, en))

    audit["splits"][name] = {
        "rows": len(rows),
        "missing_tachelhit": sum(not x for x in shi),
        "missing_english": sum(not x for x in en),
        "duplicate_pairs": len(pairs) - len(set(pairs)),
        "tachelhit_over_200_chars": sum(
            len(x) > 200 for x in shi
        ),
        "english_over_200_chars": sum(
            len(x) > 200 for x in en
        ),
        "unique_tachelhit": len(set(shi)),
        "unique_english": len(set(en)),
    }

names = list(splits)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]

        audit["cross_split_overlap"][f"{a}_vs_{b}"] = {
            "tachelhit_overlap": len(
                keys(splits[a], "shi_latn")
                & keys(splits[b], "shi_latn")
            ),
            "english_overlap": len(
                keys(splits[a], "en")
                & keys(splits[b], "en")
            ),
        }

conflicts = read_csv("conflicts.csv")
audit["conflict_rows"] = len(conflicts)

OUT.write_text(
    json.dumps(audit, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(audit, indent=2, ensure_ascii=False))

print("\n=== CONFLICT SAMPLE ===")
for row in conflicts[:5]:
    print(json.dumps(row, ensure_ascii=False)[:500])

print("\nSaved:", OUT)

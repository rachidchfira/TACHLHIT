from datasets import load_dataset
from pathlib import Path
import csv
import json
import unicodedata

ROOT = Path("datasets/processed")
ROOT.mkdir(parents=True, exist_ok=True)

REPO = (
    "abdelhaqueidali/"
    "Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset"
)

def clean(value):
    return unicodedata.normalize(
        "NFC", str(value or "").strip()
    )

def normalize_id(value):
    return clean(value).lstrip("\ufeff")

def key_for(row):
    latin = clean(row["tachelhit_latin"]).casefold()
    tifinagh = clean(row["tachelhit_tifinagh"]).casefold()
    return latin or tifinagh

print("Loading Tatoeba...")

mono = load_dataset(
    REPO, "shi_tifinagh", split="train"
)

parallel = load_dataset(
    REPO, "en_shi_tifinagh", split="train"
)

records = {}

for row in mono:
    sid = normalize_id(row["id"])
    records[sid] = {
        "source": "tatoeba",
        "source_id": sid,
        "variety": "Tachelhit",
        "tachelhit_latin": clean(row["shi_latn"]),
        "tachelhit_tifinagh": clean(row["shi_tfng"]),
        "english": "",
        "validation": "",
        "is_validated": "",
        "review_status": "unreviewed",
    }

for row in parallel:
    sid = normalize_id(row["id"])

    if sid not in records:
        records[sid] = {
            "source": "tatoeba",
            "source_id": sid,
            "variety": "Tachelhit",
            "tachelhit_latin": clean(row["shi_latn"]),
            "tachelhit_tifinagh": clean(row["shi_tfng"]),
            "english": "",
            "validation": "",
            "is_validated": "",
            "review_status": "unreviewed",
        }

    records[sid]["english"] = clean(row["en"])

print("Tatoeba IDs:", len(records))

print("Loading Amazic...")

amazic = load_dataset(
    "collectivat/amazic",
    split="train"
)

for index, row in enumerate(amazic):
    if row["tgtVar"] != "Tachelhit":
        continue

    target = clean(row["tgt_text"])

    is_tifinagh = any(
        "\u2d30" <= c <= "\u2d7f"
        for c in target
    )

    records[f"amazic-{index}"] = {
        "source": "amazic",
        "source_id": str(index),
        "variety": "Tachelhit",
        "tachelhit_latin": "" if is_tifinagh else target,
        "tachelhit_tifinagh": target if is_tifinagh else "",
        "english": clean(row["src_text"]) if row["src"] == "en" else "",
        "validation": str(row["validation"]),
        "is_validated": str(row["isValidated"]),
        "review_status": "unreviewed",
    }

# Remove exact text duplicates without destroying
# records from different source repositories.
seen = set()
unique = []
duplicates = []

for row in records.values():
    key = key_for(row)

    if not key:
        continue

    if key in seen:
        duplicates.append(row)
    else:
        seen.add(key)
        unique.append(row)

fields = [
    "source", "source_id", "variety",
    "tachelhit_latin", "tachelhit_tifinagh",
    "english", "validation", "is_validated",
    "review_status"
]

def write_csv(path, rows):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

write_csv(ROOT / "corpus_unreviewed.csv", unique)
write_csv(ROOT / "duplicates.csv", duplicates)

stats = {
    "total_unique_records": len(unique),
    "duplicates_removed": len(duplicates),
    "with_english": sum(
        bool(r["english"]) for r in unique
    ),
    "with_latin": sum(
        bool(r["tachelhit_latin"]) for r in unique
    ),
    "with_tifinagh": sum(
        bool(r["tachelhit_tifinagh"]) for r in unique
    ),
    "sources": {
        source: sum(
            r["source"] == source for r in unique
        )
        for source in ["tatoeba", "amazic"]
    },
}

(ROOT / "corpus_stats.json").write_text(
    json.dumps(stats, indent=2),
    encoding="utf-8"
)

print("\nCORPUS STATISTICS")
print(json.dumps(stats, indent=2))

print("\nSaved:")
print(ROOT / "corpus_unreviewed.csv")
print(ROOT / "duplicates.csv")
print(ROOT / "corpus_stats.json")

import csv
import json
import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

SRC = Path("datasets/reverse_translation/v1")
DST = Path("datasets/reverse_translation/v2")
DST.mkdir(parents=True, exist_ok=True)

def normalize(value):
    return " ".join(
        unicodedata.normalize("NFKC", value or "")
        .casefold().split()
    )

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

# Include original clean rows and quarantined alternative translations.
rows = read_csv(SRC / "all.csv") + read_csv(SRC / "conflicts.csv")

# Keep only valid bilingual pairs.
rows = [
    r for r in rows
    if normalize(r.get("shi_latn"))
    and normalize(r.get("en"))
]

# Deduplicate identical bilingual pairs.
unique = {}
for r in rows:
    key = (
        normalize(r["shi_latn"]),
        normalize(r["en"]),
    )
    unique.setdefault(key, r)

rows = list(unique.values())

# Union-find: group examples sharing either language side.
parent = list(range(len(rows)))
size = [1] * len(rows)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]

seen_shi = {}
seen_en = {}

for i, r in enumerate(rows):
    shi = normalize(r["shi_latn"])
    en = normalize(r["en"])

    if shi in seen_shi:
        union(i, seen_shi[shi])
    else:
        seen_shi[shi] = i

    if en in seen_en:
        union(i, seen_en[en])
    else:
        seen_en[en] = i

groups = defaultdict(list)
for i in range(len(rows)):
    groups[find(i)].append(i)

# Assign whole connected groups to splits deterministically.
# Group identity is based on content, not row order.
assigned = {
    "train": [],
    "validation": [],
    "test": [],
}

for indices in groups.values():
    group_key = min(
        normalize(rows[i]["shi_latn"]) + "\t"
        + normalize(rows[i]["en"])
        for i in indices
    )

    h = int(
        hashlib.sha256(group_key.encode()).hexdigest()[:12],
        16,
    ) / float(16 ** 12)

    split = (
        "train" if h < 0.90
        else "validation" if h < 0.95
        else "test"
    )

    for i in indices:
        row = dict(rows[i])
        row["split"] = split
        assigned[split].append(row)

fields = list(dict.fromkeys(
    key
    for row in rows
    for key in row.keys()
))
if "split" not in fields:
    fields.append("split")

for name, data in assigned.items():
    write_csv(DST / f"{name}.csv", data, fields)

all_rows = sum(assigned.values(), [])
write_csv(DST / "all.csv", all_rows, fields)

def keyset(data, field):
    return {normalize(r[field]) for r in data}

overlaps = {}

names = list(assigned)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        overlaps[f"{a}_vs_{b}"] = {
            "shi": len(
                keyset(assigned[a], "shi_latn")
                & keyset(assigned[b], "shi_latn")
            ),
            "en": len(
                keyset(assigned[a], "en")
                & keyset(assigned[b], "en")
            ),
        }

report = {
    "input_rows": len(rows),
    "connected_groups": len(groups),
    "largest_group": max(map(len, groups.values())),
    "train": len(assigned["train"]),
    "validation": len(assigned["validation"]),
    "test": len(assigned["test"]),
    "overlaps": overlaps,
    "method": "Bilingual connected-component splitting",
}

(DST / "report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))

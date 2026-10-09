from datasets import load_dataset
from pathlib import Path
import json

sources = {
    "Tachelhit Tatoeba":
        "abdelhaqueidali/Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset",
    "MT560":
        "michsethowusu/english-tachelhit_sentence-pairs_mt560",
    "Amazic":
        "collectivat/amazic",
}

report = {}

for name, repo in sources.items():
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name}")
    print(f"REPOSITORY: {repo}")

    try:
        dataset = load_dataset(repo)

        report[name] = {}

        for split_name, split in dataset.items():
            print(f"\nSplit: {split_name}")
            print(f"Rows: {len(split)}")
            print(f"Columns: {split.column_names}")

            examples = []

            for row in split.select(range(min(5, len(split)))):
                examples.append(row)
                print(json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str
                ))

            report[name][split_name] = {
                "rows": len(split),
                "columns": split.column_names,
                "examples": examples,
            }

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        report[name] = {"error": str(e)}

output = Path("datasets/audit/corpus_samples.json")

output.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        default=str
    ),
    encoding="utf-8"
)

print("\nAudit saved to:", output)

from datasets import load_dataset
from collections import Counter
import json

print("\n========== AMAZIC VARIETIES ==========")

amazic = load_dataset("collectivat/amazic", split="train")

varieties = Counter(str(x) for x in amazic["tgtVar"])
print("Variety counts:", dict(varieties))

tachelhit = amazic.filter(
    lambda row: row["tgtVar"] == "Tachelhit"
)

print("\nTachelhit records:", len(tachelhit))
print("Validation scores:",
      dict(Counter(str(x) for x in tachelhit["validation"])))
print("Marked validated:",
      dict(Counter(str(x) for x in tachelhit["isValidated"])))

print("\nTachelhit examples:")
for row in tachelhit.select(range(min(10, len(tachelhit)))):
    print(json.dumps({
        "source": row["src_text"],
        "target": row["tgt_text"],
        "validation": row["validation"],
        "isValidated": row["isValidated"]
    }, ensure_ascii=False))

print("\n========== TATOEBA CONFIGURATIONS ==========")

repo = "abdelhaqueidali/Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset"

for config in ["shi_tifinagh", "en_shi_tifinagh"]:
    print(f"\nConfiguration: {config}")

    try:
        ds = load_dataset(repo, config)

        for split_name, split in ds.items():
            print("Split:", split_name)
            print("Rows:", len(split))
            print("Columns:", split.column_names)

            for row in split.select(range(min(5, len(split)))):
                print(json.dumps(row, ensure_ascii=False))

    except Exception as e:
        print("ERROR:", type(e).__name__, str(e))

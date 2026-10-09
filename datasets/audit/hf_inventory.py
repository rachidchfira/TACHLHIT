import csv
import json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()

resources = [
    ("model", "facebook/mms-tts-shi"),
    ("model", "facebook/mms-1b-all"),
    ("model", "rachidnichan/Awal-LM"),
    ("model", "rachidnichan/awallm"),
    ("model", "linguardia/opus-mt-en-shi"),
    ("dataset", "collectivat/amazic"),
    ("dataset", "abdelhaqueidali/Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset"),
    ("dataset", "michsethowusu/english-tachelhit_sentence-pairs_mt560"),
    ("dataset", "Tamazight-NLP/IRCAM-CORPUS"),
    ("dataset", "Tamazight-NLP/Tatoeba-Translations"),
    ("dataset", "Tamazight-NLP/Pontoon-Translations"),
]

rows = []
details = []

for kind, repo in resources:
    print(f"\nInspecting {kind}: {repo}")

    try:
        if kind == "model":
            info = api.model_info(repo, files_metadata=True)
        else:
            info = api.dataset_info(repo, files_metadata=True)

        card = info.card_data
        license_name = (
            card.get("license", "unknown")
            if card else "unknown"
        )

        files = []
        total_bytes = 0

        for sibling in info.siblings or []:
            size = sibling.size or 0
            total_bytes += size
            files.append({
                "name": sibling.rfilename,
                "size_bytes": size
            })

        rows.append({
            "type": kind,
            "repository": repo,
            "status": "available",
            "license": str(license_name),
            "file_count": len(files),
            "total_gb": round(total_bytes / 1e9, 3),
            "revision": info.sha or ""
        })

        details.append({
            "type": kind,
            "repository": repo,
            "license": str(license_name),
            "files": files
        })

        print(
            f"  License: {license_name} | "
            f"Files: {len(files)} | "
            f"Listed size: {total_bytes/1e9:.3f} GB"
        )

    except Exception as exc:
        print(f"  ERROR: {exc}")
        rows.append({
            "type": kind,
            "repository": repo,
            "status": "error",
            "license": "",
            "file_count": "",
            "total_gb": "",
            "revision": ""
        })

out = Path("datasets/audit")

with (out / "inventory.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

(out / "repository_files.json").write_text(
    json.dumps(details, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nAudit completed.")
print("Inventory:", out / "inventory.csv")
print("File details:", out / "repository_files.json")

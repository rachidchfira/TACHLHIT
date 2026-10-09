#!/usr/bin/env python3
"""SoussVoice: Arabic -> Tifinagh zero-shot NLLB baseline + Tifinagh script audit.

Read-only with respect to existing datasets and adapters.

Usage (from ~/soussvoice):
  python experiments/tawnza/soussvoice_reverse_baseline.py
"""
import csv
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "facebook/nllb-200-distilled-600M"
DATA = Path("datasets/external/tawnza/clean_v3")
OUTPUT = Path("experiments/tawnza/reverse_baseline")
SAMPLE_SIZE = 80
SEED = 20261009

# High-frequency standardized IRCAM Tifinagh characters. This is a deterministic
# visual transliteration, NOT an Agadir dialect normalizer. Unmapped Tifinagh
# characters are preserved and counted rather than silently deleted.
TIF_TO_LAT = {
    "ⴰ": "a", "ⴱ": "b", "ⴳ": "g", "ⴷ": "d", "ⴹ": "ḍ",
    "ⴻ": "e", "ⴼ": "f", "ⴽ": "k", "ⵀ": "h", "ⵃ": "ḥ",
    "ⵄ": "ɛ", "ⵅ": "x", "ⵇ": "q", "ⵉ": "i", "ⵊ": "j",
    "ⵍ": "l", "ⵎ": "m", "ⵏ": "n", "ⵓ": "u", "ⵔ": "r",
    "ⵕ": "ṛ", "ⵖ": "ɣ", "ⵙ": "s", "ⵚ": "ṣ", "ⵜ": "t",
    "ⵟ": "ṭ", "ⵡ": "w", "ⵢ": "y", "ⵣ": "z", "ⵥ": "ẓ",
    "ⵛ": "c", "ⵯ": "ʷ",
}


def to_latin(text, readable=True):
    """Preserve all unknown characters; use 'ʷ' for labialization internally."""
    result = "".join(TIF_TO_LAT.get(c, c) for c in unicodedata.normalize("NFC", text or ""))
    return result.replace("ʷ", "w") if readable else result


def unknown_tifinagh(text):
    return [c for c in text or "" if "\u2d30" <= c <= "\u2d7f" and c not in TIF_TO_LAT]


def read_rows(split):
    path = DATA / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {path}")
    return pq.read_table(path, columns=["source", "target"]).to_pylist()


def main():
    print("=== SOUSSVOICE REVERSE-DIRECTION BASELINE ===", flush=True)
    print("Goal: Arabic -> Tifinagh (then Latin display), with NO training.", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.backends.mps.is_available():
        print("Apple MPS unavailable; using CPU. This may take much longer.", flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Detect writing-system coverage on all splits without changing any data.
    usage = Counter()
    unknown = Counter()
    for split in ("train", "validation", "test"):
        for row in read_rows(split):
            for char in row["target"] or "":
                if "\u2d30" <= char <= "\u2d7f":
                    usage[char] += 1
                    if char not in TIF_TO_LAT:
                        unknown[char] += 1
    print("Tifinagh character types:", len(usage))
    print("Unmapped Tifinagh character types:", len(unknown))
    for char, count in unknown.most_common(12):
        print("  UNMAPPED", char, f"U+{ord(char):04X}", count)
    (OUTPUT / "script_audit.json").write_text(
        json.dumps({
            "unique_tifinagh_chars": len(usage),
            "unmapped_tifinagh_chars": [
                {"character": c, "unicode": f"U+{ord(c):04X}", "count": n}
                for c, n in unknown.most_common()
            ],
            "interpretation": "Only a visual transliteration; dialect spelling not verified",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    validation_rows = read_rows("validation")
    if len(validation_rows) < SAMPLE_SIZE:
        raise ValueError(f"Need at least {SAMPLE_SIZE} validation rows; found {len(validation_rows)}")
    sample = random.Random(SEED).sample(validation_rows, SAMPLE_SIZE)
    print("\nLoading cached NLLB base checkpoint...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, src_lang="arb_Arab", tgt_lang="tzm_Tfng")
    target_id = tokenizer.convert_tokens_to_ids("tzm_Tfng")
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer does not recognize tzm_Tfng target token")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE).to(device).eval()
    outputs = []
    torch.set_num_threads(4)
    for start in range(0, len(sample), 4):
        batch = sample[start : start + 4]
        inputs = tokenizer(
            [row["source"] for row in batch],
            return_tensors="pt", padding=True, truncation=True, max_length=128,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                num_beams=3,
                max_new_tokens=80,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"Translated {min(start + len(batch), len(sample))}/{len(sample)}", flush=True)

    references = [row["target"] for row in sample]
    score = sacrebleu.corpus_chrf(outputs, [references]).score
    predictions_file = OUTPUT / "predictions.csv"
    with predictions_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arabic_input", "reference_tifinagh", "reference_latin",
                    "predicted_tifinagh", "predicted_latin", "unknown_predicted_chars"])
        for row, pred in zip(sample, outputs):
            w.writerow([
                row["source"], row["target"], to_latin(row["target"]),
                pred, to_latin(pred), "".join(unknown_tifinagh(pred)),
            ])

    report = {
        "task": "Zero-shot Arabic -> Central Atlas Tamazight (Tifinagh) proxy baseline",
        "model": BASE,
        "source_lang": "arb_Arab",
        "target_lang": "tzm_Tfng",
        "sample_size": len(sample),
        "sample_seed": SEED,
        "validation_corpus": str(DATA / "validation.parquet"),
        "chrf": round(score, 3),
        "note": "Mixed-variety corpus; chrF does not measure authentic Souss speech or conversational fluency.",
        "predictions_csv": str(predictions_file),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n======= REVERSE BASELINE RESULTS =======", flush=True)
    print("Arabic -> Tifinagh examples:", len(sample), flush=True)
    print(f"chrF: {score:.2f}", flush=True)
    print("Saved:", predictions_file, flush=True)
    print("\nFIRST EIGHT EXAMPLES:", flush=True)
    for i, (row, pred) in enumerate(zip(sample[:8], outputs[:8]), 1):
        print(f"\n{i}. AR:       {row['source']}")
        print("   REF LAT:  ", to_latin(row["target"]))
        print("   MODEL LAT:", to_latin(pred))
        print("   MODEL TIF:", pred)
    print("\nNo model adapters or source datasets were modified.", flush=True)


if __name__ == "__main__":
    main()

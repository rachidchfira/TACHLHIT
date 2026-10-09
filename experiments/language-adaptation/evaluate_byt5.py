import csv
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CHECKPOINT = Path(
    "experiments/language-adaptation/byt5_runs/v1/checkpoint-500"
)
VALIDATION = Path("datasets/reverse_translation/v2/validation.csv")
OUTPUT = Path(
    "experiments/language-adaptation/byt5_runs/v1/translation_samples.csv"
)

assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
assert VALIDATION.exists(), f"Missing dataset: {VALIDATION}"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
model.to(device)
model.eval()

with VALIDATION.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

rng = random.Random(2026)
samples = rng.sample(rows, min(20, len(rows)))

results = []

for i, row in enumerate(samples, 1):
    source = row["shi_latn"]
    reference = row["en"]

    inputs = tokenizer(
        "Translate Tachelhit to English: " + source,
        return_tensors="pt",
        max_length=160,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            num_beams=1,
            do_sample=False,
        )

    prediction = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )

    result = {
        "tachelhit": source,
        "reference": reference,
        "prediction": prediction,
    }
    results.append(result)

    print(f"\n--- Example {i}/20 ---", flush=True)
    print("Tachelhit:", source, flush=True)
    print("Expected: ", reference, flush=True)
    print("Model:    ", prediction, flush=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["tachelhit", "reference", "prediction"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved results:", OUTPUT, flush=True)

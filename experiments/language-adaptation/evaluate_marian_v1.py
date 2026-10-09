import csv
import json
import random
import time
from pathlib import Path

import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("experiments/language-adaptation")
FINETUNED = ROOT / "marian_runs/v1/best_model"
BASELINE = "Helsinki-NLP/opus-mt-ber-en"
TEST = Path("datasets/reverse_translation/v2/test.csv")
OUT = ROOT / "marian_runs/v1/evaluation"
OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
MAX_INPUT_TOKENS = 160
MAX_NEW_TOKENS = 80
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device, flush=True)

with TEST.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

sources = [r["shi_latn"] for r in rows]
references = [r["en"] for r in rows]

print("Test examples:", len(rows), flush=True)

def predict(model_name, label):
    print(f"\nLoading {label}: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    ).to(device)
    model.eval()

    predictions = []
    start = time.time()

    for offset in range(0, len(sources), BATCH_SIZE):
        batch = sources[offset:offset + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                do_sample=False,
            )

        predictions.extend(
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
        )

        completed = min(offset + BATCH_SIZE, len(sources))
        if completed % 80 == 0 or completed == len(sources):
            print(
                f"{label}: {completed}/{len(sources)}",
                flush=True
            )

    elapsed = round(time.time() - start, 1)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return predictions, elapsed

baseline_predictions, baseline_seconds = predict(
    BASELINE, "Original Marian"
)

finetuned_predictions, finetuned_seconds = predict(
    str(FINETUNED), "Fine-tuned Marian"
)

def calculate_metrics(predictions):
    return {
        "bleu": round(
            sacrebleu.corpus_bleu(
                predictions, [references]
            ).score, 4
        ),
        "chrf": round(
            sacrebleu.corpus_chrf(
                predictions, [references]
            ).score, 4
        ),
    }

baseline_metrics = calculate_metrics(baseline_predictions)
finetuned_metrics = calculate_metrics(finetuned_predictions)

metrics = {
    "test_examples": len(rows),
    "seed": SEED,
    "device": str(device),
    "generation": {
        "batch_size": BATCH_SIZE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": 4,
    },
    "versions": {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "sacrebleu": sacrebleu.__version__,
    },
    "baseline": {
        **baseline_metrics,
        "seconds": baseline_seconds,
    },
    "finetuned": {
        **finetuned_metrics,
        "seconds": finetuned_seconds,
    },
}

with (OUT / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

fieldnames = [
    "example_id",
    "shi_latn",
    "reference_en",
    "baseline_en",
    "finetuned_en",
]

comparison = []
for row, base, tuned in zip(
    rows, baseline_predictions, finetuned_predictions
):
    comparison.append({
        "example_id": row["example_id"],
        "shi_latn": row["shi_latn"],
        "reference_en": row["en"],
        "baseline_en": base,
        "finetuned_en": tuned,
    })

def save_csv(path, data):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

save_csv(OUT / "test_predictions.csv", comparison)

sample = random.sample(
    comparison, min(150, len(comparison))
)
save_csv(OUT / "native_review_sample.csv", sample)

report = f"""# SoussVoice Marian V1 Evaluation

Test examples: {len(rows)}

| Model | BLEU | chrF |
|---|---:|---:|
| Original Marian | {baseline_metrics['bleu']} | {baseline_metrics['chrf']} |
| Fine-tuned Marian | {finetuned_metrics['bleu']} | {finetuned_metrics['chrf']} |

## Runtime
- Original Marian: {baseline_seconds} seconds
- Fine-tuned Marian: {finetuned_seconds} seconds

## Interpretation
BLEU and chrF measure similarity to reference translations.
They do not establish semantic correctness or dialect fidelity.

Native-speaker review is required before using these predictions
as verified training labels.
"""

(OUT / "evaluation_report.md").write_text(
    report, encoding="utf-8"
)

print("\n===== FINAL RESULTS =====")
print(json.dumps(metrics, indent=2))
print("\nSaved results to:", OUT)

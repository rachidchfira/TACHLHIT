import csv
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "linguardia/opus-mt-en-shi"

sentences = [
    "Hello, how are you?",
    "What is your name?",
    "My name is Rachid.",
    "Where are you going?",
    "I am going home.",
    "I don't understand.",
    "Thank you very much.",
    "How old are you?",
    "What are you doing?",
    "I want to learn Tachelhit.",
]

print("Loading Tachelhit translation model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    extra_special_tokens={}
)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()

results = []

for index, english in enumerate(sentences, 1):
    inputs = tokenizer(
        english,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
    )

    translation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(f"\n{index}. English: {english}")
    print(f"   Tachelhit: {translation}", flush=True)

    results.append({
        "id": index,
        "english": english,
        "model_translation": translation,
        "native_approved": "",
        "native_correction": "",
        "notes": "",
    })

output_path = Path(
    "experiments/translation/translation_results.csv"
)

with output_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {output_path}")

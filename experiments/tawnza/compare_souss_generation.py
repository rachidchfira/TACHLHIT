#!/usr/bin/env python3
"""Compare an existing English->Tachelhit model on five everyday meanings.
No fine-tuning or changes to existing SoussVoice checkpoints.
"""
import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "linguardia/opus-mt-en-shi"
SENTENCES = [
    ("أريد أن أشرب الماء", "I want to drink water."),
    ("أين ستذهب غدا؟", "Where will you go tomorrow?"),
    ("أمي تحضر الطعام في المطبخ", "My mother is preparing food in the kitchen."),
    ("لماذا لم تأت البارحة؟", "Why didn't you come yesterday?"),
    ("أنا أتعلم اللغة الأمازيغية", "I am learning the Amazigh language."),
]

print(f"Loading {MODEL_ID} (first run may download weights)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

inputs = tokenizer(
    [">>shi<< " + english for _, english in SENTENCES],
    return_tensors="pt", padding=True, truncation=True, max_length=96,
).to(device)
with torch.inference_mode():
    tokens = model.generate(**inputs, num_beams=4, max_new_tokens=80)
outputs = tokenizer.batch_decode(tokens, skip_special_tokens=True)

out = Path("experiments/tawnza/generation_comparison_en_shi.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["arabic_intent", "english_intent", "english_to_tachelhit_output"])
    for (arabic, english), output in zip(SENTENCES, outputs):
        writer.writerow([arabic, english, output])

print("\n===== ENGLISH -> TACHELHIT COMPARISON =====")
for i, ((arabic, english), output) in enumerate(zip(SENTENCES, outputs), 1):
    print(f"\n{i}. Arabic: {arabic}\n   English: {english}\n   Model: {output}")
print(f"\nSaved: {out}\nNo adapters were modified.")

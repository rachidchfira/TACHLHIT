import csv
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA = Path("datasets/reverse_translation/v2/train.csv")
OUT = Path("experiments/language-adaptation/byt5_runs/overfit32")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
STEPS = 600
CHECK_EVERY = 100
MAX_SOURCE = 160
MAX_TARGET = 128

random.seed(SEED)
torch.manual_seed(SEED)

with DATA.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Prefer short examples for a fast, interpretable diagnostic.
eligible = [
    r for r in rows
    if r["shi_latn"].strip()
    and r["en"].strip()
    and len(r["shi_latn"]) <= 65
    and len(r["en"]) <= 85
]

# One source and one target per selected example.
# Avoid conflicting references within this diagnostic.
random.shuffle(eligible)
selected = []
seen_sources = set()
seen_targets = set()

for row in eligible:
    src = row["shi_latn"].strip()
    tgt = row["en"].strip()

    if src.casefold() in seen_sources:
        continue
    if tgt.casefold() in seen_targets:
        continue

    selected.append((src, tgt))
    seen_sources.add(src.casefold())
    seen_targets.add(tgt.casefold())

    if len(selected) == 32:
        break

assert len(selected) == 32, "Not enough eligible examples"

with (OUT / "examples.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["shi_latn", "en"])
    writer.writerows(selected)

print("Selected:", len(selected), "training pairs", flush=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def encode(src, tgt):
    inputs = tokenizer(
        "Translate Tachelhit to English: " + src,
        max_length=MAX_SOURCE,
        truncation=True,
        return_tensors="pt",
    )
    targets = tokenizer(
        text_target=tgt,
        max_length=MAX_TARGET,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "labels": targets["input_ids"].to(device),
    }

@torch.no_grad()
def inspect(step):
    model.eval()
    correct = 0
    predictions = []

    for src, tgt in selected:
        batch = tokenizer(
            "Translate Tachelhit to English: " + src,
            max_length=MAX_SOURCE,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        ids = model.generate(
            **batch,
            max_new_tokens=100,
            num_beams=1,
            do_sample=False,
        )
        pred = tokenizer.decode(
            ids[0], skip_special_tokens=True
        ).strip()

        if pred.casefold() == tgt.casefold():
            correct += 1

        predictions.append((src, tgt, pred))

    with (OUT / f"predictions_{step}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["shi_latn", "expected", "predicted"])
        writer.writerows(predictions)

    print(
        f"\nStep {step}: exact translations "
        f"{correct}/32",
        flush=True,
    )

    for src, tgt, pred in predictions[:5]:
        print("SHI:", src, flush=True)
        print("REF:", tgt, flush=True)
        print("OUT:", pred, flush=True)
        print("---", flush=True)

    model.train()
    return correct

start = time.time()
model.train()

for step in range(1, STEPS + 1):
    src, tgt = selected[(step - 1) % 32]
    batch = encode(src, tgt)

    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step}/{STEPS} | "
            f"loss {loss.item():.4f} | "
            f"elapsed {(time.time()-start)/60:.1f} min",
            flush=True,
        )

    if step % CHECK_EVERY == 0:
        correct = inspect(step)

        if correct == 32:
            print("SUCCESS: 32/32 memorized", flush=True)
            break

print("\nFinished.", flush=True)
print("Results:", OUT, flush=True)

import csv
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("experiments/language-adaptation/byt5_runs/overfit32")
DATA = ROOT / "examples.csv"
OUT = ROOT / "v2"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/byt5-small"
BATCH_SIZE = 8
EPOCHS = 60
LR = 3e-5
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

with DATA.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 32, f"Expected 32 examples, got {len(rows)}"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def make_batch(items):
    sources = [
        "Translate Tachelhit to English: " + r["shi_latn"]
        for r in items
    ]
    targets = [r["en"] for r in items]

    enc = tok(
        sources,
        padding=True,
        truncation=True,
        max_length=160,
        return_tensors="pt",
    )

    tgt = tok(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    labels = tgt["input_ids"]
    labels[labels == tok.pad_token_id] = -100

    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": labels.to(device),
    }

@torch.no_grad()
def evaluate(epoch):
    model.eval()
    correct = 0
    predictions = []

    for row in rows:
        src = row["shi_latn"]
        ref = row["en"]

        inputs = tok(
            "Translate Tachelhit to English: " + src,
            return_tensors="pt",
            max_length=160,
            truncation=True,
        ).to(device)

        ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            num_beams=1,
        )

        pred = tok.decode(
            ids[0], skip_special_tokens=True
        ).strip()

        correct += pred.casefold() == ref.casefold()
        predictions.append((src, ref, pred))

    with (OUT / f"predictions_epoch_{epoch}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["shi_latn", "reference", "prediction"])
        writer.writerows(predictions)

    print(
        f"\nEpoch {epoch}: exact matches {correct}/32",
        flush=True,
    )

    for src, ref, pred in predictions[:5]:
        print(f"SHI: {src}")
        print(f"REF: {ref}")
        print(f"OUT: {pred}")
        print("---")

    model.train()
    return correct

start = time.time()

for epoch in range(1, EPOCHS + 1):
    random.shuffle(rows)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses = []

    for start_idx in range(0, 32, BATCH_SIZE):
        batch = make_batch(rows[start_idx:start_idx + BATCH_SIZE])

        loss = model(**batch).loss
        (loss / 4).backward()
        losses.append(loss.item())

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"| mean loss {sum(losses)/len(losses):.4f} "
            f"| elapsed {(time.time()-start)/60:.1f} min",
            flush=True,
        )

    if epoch % 10 == 0:
        correct = evaluate(epoch)

        if correct == 32:
            print("SUCCESS: all 32 memorized", flush=True)
            break

print("\nFinished. Results:", OUT)

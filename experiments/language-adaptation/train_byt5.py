import csv
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("datasets/reverse_translation/v2")
OUT = Path("experiments/language-adaptation/byt5_runs/v1")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "google/byt5-small"
STEPS = 500
SAVE_EVERY = 100
VALIDATION_EXAMPLES = 100
MAX_SOURCE_LENGTH = 160
MAX_TARGET_LENGTH = 128
LR = 3e-5
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

def read_csv(name):
    with (ROOT / name).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

train = read_csv("train.csv")
validation = read_csv("validation.csv")

assert train and validation
assert all(r["shi_latn"] and r["en"] for r in train)
assert all(r["shi_latn"] and r["en"] for r in validation)

rng = random.Random(SEED)
rng.shuffle(train)
validation = rng.sample(
    validation,
    min(VALIDATION_EXAMPLES, len(validation))
)

print("Training pairs:", len(train), flush=True)
print("Validation sample:", len(validation), flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR
)

def encode(row):
    src = tokenizer(
        "Translate Tachelhit to English: " + row["shi_latn"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    tgt = tokenizer(
        text_target=row["en"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": src["input_ids"].to(device),
        "attention_mask": src["attention_mask"].to(device),
        "labels": tgt["input_ids"].to(device),
    }

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for row in validation:
        batch = encode(row)
        loss = model(**batch).loss
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def save(step, val_loss):
    path = OUT / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "metrics.json").write_text(
        json.dumps(
            {"step": step, "validation_loss": val_loss},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved:", path, flush=True)

start = time.time()
model.train()
history = []
best_loss = float("inf")
best_step = 0

for step in range(1, STEPS + 1):
    row = train[(step - 1) % len(train)]
    batch = encode(row)

    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 20 == 0:
        print(
            f"Step {step}/{STEPS} "
            f"| train loss {loss.item():.4f} "
            f"| elapsed {(time.time()-start)/60:.1f} min",
            flush=True,
        )

    if step % SAVE_EVERY == 0:
        val_loss = evaluate()
        print(
            f"Validation loss at step {step}: "
            f"{val_loss:.4f}",
            flush=True,
        )
        history.append({
            "step": step,
            "validation_loss": val_loss,
        })
        save(step, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_step = step

        (OUT / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

print("\nTraining complete.")
print("Best step:", best_step)
print("Best validation loss:", round(best_loss, 4))
print("Checkpoints:", OUT)

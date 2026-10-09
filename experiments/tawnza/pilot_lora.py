import csv
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model

BASE = "facebook/nllb-200-distilled-600M"
DATA = Path("datasets/external/tawnza/clean_v2")
OUT = Path("experiments/tawnza/lora_pilot")
OUT.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)
torch.manual_seed(42)
torch.set_num_threads(4)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device, flush=True)

train_rows = pq.read_table(DATA / "train.parquet").to_pylist()
test_rows = pq.read_table(DATA / "test.parquet").to_pylist()

rng.shuffle(train_rows)
train_rows = train_rows[:3000]

# Reproduce the previous 30-sentence evaluation sample.
evaluation = random.Random(42).sample(test_rows, 30)

tok = AutoTokenizer.from_pretrained(
    BASE,
    src_lang="tzm_Tfng",
    tgt_lang="arb_Arab",
)
arabic_id = tok.convert_tokens_to_ids("arb_Arab")

model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = get_peft_model(
    model,
    LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
).to(device)

model.print_trainable_parameters()

def translate(rows):
    model.eval()
    model.config.use_cache = True
    predictions = []

    for start in range(0, len(rows), 5):
        batch = rows[start:start + 5]
        encoded = tok(
            [r["target"] for r in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **encoded,
                forced_bos_token_id=arabic_id,
                num_beams=3,
                max_new_tokens=80,
            )

        predictions.extend(
            tok.batch_decode(output, skip_special_tokens=True)
        )

    return predictions

def score(predictions):
    refs = [r["source"] for r in evaluation]
    return sacrebleu.corpus_chrf(
        predictions, [refs]
    ).score

print("\nEvaluating BEFORE training...", flush=True)
before = translate(evaluation)
before_score = score(before)
print(f"BEFORE chrF: {before_score:.2f}", flush=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,
)

model.train()
model.config.use_cache = False

MAX_STEPS = 150
MAX_SECONDS = 900
ACCUM = 2

optimizer.zero_grad(set_to_none=True)
start_time = time.monotonic()
steps = 0
loss_sum = 0.0

print("\nStarting limited LoRA training...", flush=True)

for i in range(MAX_STEPS * ACCUM):
    if time.monotonic() - start_time > MAX_SECONDS:
        print("Time limit reached.", flush=True)
        break

    r = train_rows[i % len(train_rows)]

    encoded = tok(
        r["target"],
        return_tensors="pt",
        truncation=True,
        max_length=96,
    ).to(device)

    labels = tok(
        text_target=r["source"],
        return_tensors="pt",
        truncation=True,
        max_length=96,
    )["input_ids"].to(device)

    output = model(**encoded, labels=labels)
    loss = output.loss / ACCUM
    loss.backward()
    loss_sum += loss.item() * ACCUM

    if (i + 1) % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1

        if steps % 10 == 0:
            print(
                f"Step {steps}/{MAX_STEPS} | "
                f"Avg loss: {loss_sum / (10 * ACCUM):.4f} | "
                f"Elapsed: {(time.monotonic() - start_time)/60:.1f} min",
                flush=True
            )
            loss_sum = 0.0

model.save_pretrained(OUT / "adapter")
tok.save_pretrained(OUT / "adapter")

print("\nEvaluating AFTER training...", flush=True)
after = translate(evaluation)
after_score = score(after)

with (OUT / "comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "tifinagh", "arabic_reference",
        "before", "after"
    ])
    for row, old, new in zip(evaluation, before, after):
        writer.writerow([
            row["target"], row["source"], old, new
        ])

print("\n========== RESULTS ==========")
print(f"Steps completed: {steps}")
print(f"BEFORE chrF: {before_score:.2f}")
print(f"AFTER  chrF: {after_score:.2f}")
print(f"CHANGE: {after_score - before_score:+.2f}")
print(f"Saved: {OUT / 'comparison.csv'}")

print("\nFIRST FIVE COMPARISONS:")
for row, old, new in list(zip(evaluation, before, after))[:5]:
    print("\nReference:", row["source"])
    print("Before:   ", old)
    print("After:    ", new)

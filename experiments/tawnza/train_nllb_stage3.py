#!/usr/bin/env python3
"""SoussVoice: budgeted NLLB LoRA adaptation on TAWNZA.

Reuses V2 adapter, trains on a reproducible subset of distinct training
pairs with 2 passes, measures chrF/BLEU on unseen validation episodes,
and saves the best checkpoint and human-readable comparisons.
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "facebook/nllb-200-distilled-600M"
DATA = Path("datasets/external/tawnza/clean_v2")
OLD = Path("experiments/tawnza/lora_pilot_v2/adapter")
OUT = Path("experiments/tawnza/stage3")
BEST = OUT / "best_adapter"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 6000
VAL_SIZE = 180
EPOCHS = 2
BATCH_SIZE = 4
ACCUM = 2
LR = 1e-4
TRAIN_TIME_LIMIT_SECONDS = 1800  # 30 minutes, evaluations included
EVAL_EVERY_STEPS = 375
SEED = 2026

if not (OLD / "adapter_config.json").is_file():
    raise SystemExit(f"Missing V2 adapter: {OLD}")
for split in ("train", "validation"):
    if not (DATA / f"{split}.parquet").is_file():
        raise SystemExit(f"Missing data: {DATA / (split + '.parquet')}")

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"DEVICE: {device}", flush=True)

all_train = pq.read_table(DATA / "train.parquet", columns=["source", "target"]).to_pylist()
all_val = pq.read_table(DATA / "validation.parquet", columns=["source", "target"]).to_pylist()
train = random.Random(SEED).sample(all_train, min(TRAIN_SIZE, len(all_train)))
val = random.Random(909).sample(all_val, min(VAL_SIZE, len(all_val)))
del all_train, all_val
print(f"TRAIN UNIQUE PAIRS: {len(train):,} | VAL PAIRS: {len(val)} | EPOCHS: {EPOCHS}", flush=True)

print("Loading cached NLLB and the V2 LoRA adapter...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, src_lang="tzm_Tfng", tgt_lang="arb_Arab")
base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, str(OLD), is_trainable=True).to(device)
ar_id = tok.convert_tokens_to_ids("arb_Arab")
assert ar_id != tok.unk_token_id, "Arabic language ID not recognized"
trainable = [p for p in model.parameters() if p.requires_grad]
assert trainable, "No trainable parameters"
print("TRAINABLE PARAMETERS:", sum(p.numel() for p in trainable), flush=True)

refs = [r["source"] for r in val]

def predict(rows):
    model.eval()
    model.config.use_cache = True
    results = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        src = tok(
            [r["target"] for r in batch], return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **src, forced_bos_token_id=ar_id, num_beams=3,
                max_new_tokens=80,
            )
        results.extend(tok.batch_decode(output, skip_special_tokens=True))
    return results

def score(predictions):
    return {
        "chrf": round(sacrebleu.corpus_chrf(predictions, [refs]).score, 4),
        "bleu": round(sacrebleu.corpus_bleu(predictions, [refs]).score, 4),
    }

def persist_comparison(filename, predictions, baseline):
    with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tifinagh", "arabic_reference", "baseline_v2", "stage3"])
        for row, before, after in zip(val, baseline, predictions):
            writer.writerow([row["target"], row["source"], before, after])

print("Evaluating V2 before additional training (same validation examples)...", flush=True)
before = predict(val)
base_score = score(before)
print("BASELINE V2:", base_score, flush=True)
model.save_pretrained(BEST)
tok.save_pretrained(BEST)

optimizer = torch.optim.AdamW(trainable, lr=LR)
started = time.monotonic()
optimizer_steps = 0
examples_presented = 0
best_chrf = base_score["chrf"]
best_at_step = 0
best_pred = list(before)
checkpoints = [{"step": 0, "examples": 0, **base_score}]
last_loss = None
print("Starting Stage 3 (30-min budget including periodic validation)...", flush=True)

try:
    for epoch in range(1, EPOCHS + 1):
        shuffled = list(train)
        random.Random(SEED + epoch).shuffle(shuffled)
        for offset in range(0, len(shuffled), BATCH_SIZE * ACCUM):
            if time.monotonic() - started >= TRAIN_TIME_LIMIT_SECONDS:
                print("TIME BUDGET REACHED.", flush=True)
                raise StopIteration

            step_rows = shuffled[offset:offset + BATCH_SIZE * ACCUM]
            micro_batches = [step_rows[k:k + BATCH_SIZE]
                             for k in range(0, len(step_rows), BATCH_SIZE)]
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            model.train()
            model.config.use_cache = False
            for batch in micro_batches:
                sources = tok(
                    [r["target"] for r in batch], return_tensors="pt",
                    padding=True, truncation=True, max_length=128,
                ).to(device)
                target = tok(
                    text_target=[r["source"] for r in batch], return_tensors="pt",
                    padding=True, truncation=True, max_length=128,
                )["input_ids"].to(device)
                target[target == tok.pad_token_id] = -100
                loss = model(**sources, labels=target).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Nonfinite loss at step {optimizer_steps}")
                (loss / len(micro_batches)).backward()
                total_loss += loss.item() / len(micro_batches)
                examples_presented += len(batch)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer_steps += 1
            last_loss = round(total_loss, 4)

            if optimizer_steps % 75 == 0:
                print(
                    f"step={optimizer_steps} epoch={epoch} "
                    f"examples={examples_presented} loss={last_loss:.3f} "
                    f"minutes={(time.monotonic() - started)/60:.1f}", flush=True,
                )

            if optimizer_steps % EVAL_EVERY_STEPS == 0:
                print(f"Validating after step {optimizer_steps}...", flush=True)
                preds = predict(val)
                metrics = score(preds)
                checkpoints.append({"step": optimizer_steps,
                                    "examples": examples_presented, **metrics})
                print(f"VALIDATION: {metrics}", flush=True)
                if metrics["chrf"] > best_chrf:
                    best_chrf = metrics["chrf"]
                    best_at_step = optimizer_steps
                    best_pred = preds
                    model.save_pretrained(BEST)
                    tok.save_pretrained(BEST)
                    print("SAVED NEW BEST ADAPTER", flush=True)

except (KeyboardInterrupt, StopIteration) as exc:
    print(f"Training stopped: {type(exc).__name__}", flush=True)

# Always compare the current adapter with best and preserve both.
print("Evaluating final current adapter...", flush=True)
final_pred = predict(val)
final_score = score(final_pred)
checkpoints.append({"step": optimizer_steps, "examples": examples_presented,
                    "final": True, **final_score})
model.save_pretrained(OUT / "last_adapter")
if final_score["chrf"] > best_chrf:
    best_chrf = final_score["chrf"]
    best_at_step = optimizer_steps
    best_pred = final_pred
    model.save_pretrained(BEST)
    tok.save_pretrained(BEST)

persist_comparison("best_comparison.csv", best_pred, before)
persist_comparison("last_comparison.csv", final_pred, before)

result = {
    "base": BASE,
    "initial_adapter": str(OLD),
    "training_unique_pairs": len(train),
    "training_example_presentations": examples_presented,
    "optimizer_steps": optimizer_steps,
    "elapsed_minutes": round((time.monotonic() - started)/60, 2),
    "baseline_v2": base_score,
    "last": final_score,
    "best_chrf": best_chrf,
    "best_at_step": best_at_step,
    "history": checkpoints,
    "best_adapter": str(BEST),
}
(OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n========== STAGE 3 RESULTS ==========", flush=True)
print(f"Optimizer steps: {optimizer_steps}")
print(f"Training sentence presentations: {examples_presented}")
print(f"Baseline V2 chrF: {base_score['chrf']:.2f}")
print(f"Last chrF: {final_score['chrf']:.2f}")
print(f"Best chrF: {best_chrf:.2f} (step {best_at_step})")
print(f"Delta vs baseline: {best_chrf - base_score['chrf']:+.2f}")
print("Files:", OUT)
print("\nEXAMPLES FROM BEST ADAPTER:")
for row, old, new in list(zip(val, before, best_pred))[:6]:
    print("\nTifinagh:", row["target"])
    print("Reference:", row["source"])
    print("V2:", old)
    print("Stage3:", new)

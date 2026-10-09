#!/usr/bin/env python3
"""Continue SoussVoice TAWNZA NLLB LoRA pilot (V2).

Inputs: existing V1 PEFT adapter and clean_v2 train/validation Parquet files.
Outputs: separate V2 adapter, 100-example validation comparison, metrics.
No modifications to existing V1 adapter or datasets.
"""
import csv
import json
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
OLD = Path("experiments/tawnza/lora_pilot/adapter")
OUT = Path("experiments/tawnza/lora_pilot_v2")

SEED = 42
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 200
TIME_LIMIT_SECONDS = 900
NUM_TRAIN_EXAMPLES = BATCH_SIZE * GRAD_ACCUM * MAX_STEPS
NUM_EVAL_EXAMPLES = 100


def main():
    assert (OLD / "adapter_config.json").exists(), f"Missing V1 adapter: {OLD}"
    for split in ("train", "validation"):
        assert (DATA / f"{split}.parquet").exists(), f"Missing {split} dataset"
    OUT.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train = pq.read_table(DATA / "train.parquet", columns=["source", "target"]).to_pylist()
    validation = pq.read_table(DATA / "validation.parquet", columns=["source", "target"]).to_pylist()
    random.Random(2026).shuffle(train)
    train = train[:NUM_TRAIN_EXAMPLES]
    evaluation = random.Random(SEED).sample(validation, min(NUM_EVAL_EXAMPLES, len(validation)))
    print(f"Training pool: {len(train)} | Validation sample: {len(evaluation)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, src_lang="tzm_Tfng", tgt_lang="arb_Arab")
    arabic_id = tok.convert_tokens_to_ids("arb_Arab")
    assert isinstance(arabic_id, int) and arabic_id != tok.unk_token_id, "Unknown Arabic language token"

    print("Loading existing NLLB and V1 adapter...", flush=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base, str(OLD), is_trainable=True).to(device)

    def translate(rows):
        model.eval()
        model.config.use_cache = True
        predictions = []
        for start in range(0, len(rows), 4):
            group = rows[start : start + 4]
            inputs = tok(
                [r["target"] for r in group],
                return_tensors="pt", padding=True, truncation=True, max_length=128,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, forced_bos_token_id=arabic_id, num_beams=3, max_new_tokens=80,
                )
            predictions.extend(tok.batch_decode(generated, skip_special_tokens=True))
        return predictions

    def chrf(predictions):
        return sacrebleu.corpus_chrf(
            predictions, [[r["source"] for r in evaluation]]
        ).score

    print("Evaluating V1 on 100 *validation* examples...", flush=True)
    before = translate(evaluation)
    before_score = chrf(before)
    print(f"BEFORE chrF: {before_score:.2f}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    started = time.monotonic()
    steps, examples_seen, last_loss = 0, 0, None
    model.train()
    model.config.use_cache = False
    print("Training V2 (max 200 steps, max 15 minutes in training loop)...", flush=True)

    for step in range(MAX_STEPS):
        if time.monotonic() - started >= TIME_LIMIT_SECONDS:
            print("Time limit reached; saving checkpoint.", flush=True)
            break
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro in range(GRAD_ACCUM):
            offset = (step * GRAD_ACCUM + micro) * BATCH_SIZE
            batch = train[offset : offset + BATCH_SIZE]
            if not batch:
                break
            encoded = tok(
                [r["target"] for r in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=96,
            ).to(device)
            labels = tok(
                text_target=[r["source"] for r in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=96,
            )["input_ids"].to(device)
            labels[labels == tok.pad_token_id] = -100
            loss = model(**encoded, labels=labels).loss
            (loss / GRAD_ACCUM).backward()
            loss_total += float(loss.item()) / GRAD_ACCUM
            examples_seen += len(batch)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        steps += 1
        last_loss = loss_total
        if steps % 20 == 0:
            print(
                f"Step {steps}/{MAX_STEPS} | Loss {loss_total:.3f} | "
                f"Examples {examples_seen} | Elapsed {(time.monotonic()-started)/60:.1f} min",
                flush=True,
            )

    print("Saving V2 adapter...", flush=True)
    model.save_pretrained(OUT / "adapter")
    tok.save_pretrained(OUT / "adapter")
    print("Evaluating V2...", flush=True)
    after = translate(evaluation)
    after_score = chrf(after)

    with (OUT / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tifinagh", "arabic_reference", "before", "after"])
        for row, old, new in zip(evaluation, before, after):
            writer.writerow([row["target"], row["source"], old, new])

    metrics = {
        "steps": steps,
        "examples_seen": examples_seen,
        "validation_examples": len(evaluation),
        "before_chrf": before_score,
        "after_chrf": after_score,
        "change": after_score - before_score,
        "last_train_loss": last_loss,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n========== V2 RESULTS ==========", flush=True)
    print(f"Steps: {steps}")
    print(f"Examples seen: {examples_seen}")
    print(f"BEFORE chrF: {before_score:.2f}")
    print(f"AFTER chrF: {after_score:.2f}")
    print(f"CHANGE: {after_score-before_score:+.2f}")
    print(f"Saved: {OUT / 'comparison.csv'}")
    print("\nFIRST FIVE COMPARISONS:")
    for row, old, new in list(zip(evaluation, before, after))[:5]:
        print(f"\nREFERENCE: {row['source']}")
        print(f"BEFORE:    {old}")
        print(f"AFTER:     {new}")


if __name__ == "__main__":
    main()

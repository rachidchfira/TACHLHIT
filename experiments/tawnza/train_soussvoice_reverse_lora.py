#!/usr/bin/env python3
"""SoussVoice: bounded Arabic -> Tifinagh NLLB LoRA experiment.

Starts from the frozen NLLB base (NOT from the forward translation adapter),
uses the same 80 validation examples as soussvoice_reverse_baseline.py,
trains on TAWNZA clean_v3/train only, and preserves previous experiments.

Run from ~/soussvoice:
    caffeinate -i python experiments/tawnza/train_soussvoice_reverse_lora.py
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/external/tawnza/clean_v3"
PREV = ROOT / "experiments/tawnza/reverse_baseline/predictions.csv"
OUT = ROOT / "experiments/tawnza/reverse_lora_v1"
BASE = "facebook/nllb-200-distilled-600M"
SEED = 20261009
SAMPLE_SIZE = 80
TRAIN_UNIQUE = 4000
EPOCHS = 2
MICROBATCH = 2
ACCUM = 4
MAX_STEPS = 1000
MAX_TRAIN_SECONDS = 30 * 60
CHECK_EVERY = 200
LR = 1e-4
MAX_INPUT = 112
MAX_TARGET = 112


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def get_rows(split):
    return pq.read_table(DATA / f"{split}.parquet", columns=["source", "target"]).to_pylist()


def compare_prior(sample):
    if not PREV.is_file():
        raise FileNotFoundError(f"Missing reverse baseline: {PREV}. Complete reverse baseline first.")
    with PREV.open(encoding="utf-8", newline="") as file:
        prev = list(csv.DictReader(file))
    if len(prev) != len(sample):
        raise RuntimeError("Previous 80-row sample length does not match; refusing incomparable benchmark.")
    for i, (row, ref) in enumerate(zip(sample, prev), 1):
        if row["source"] != ref["arabic_input"] or row["target"] != ref["reference_tifinagh"]:
            raise RuntimeError(f"Reverse baseline differs at row {i}. Do not compare these scores.")
    return [r["predicted_tifinagh"] for r in prev]


def main():
    if (OUT / "results.json").exists():
        raise RuntimeError(f"Completed results already exist at {OUT}; refusing to overwrite.")
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable. Avoid a long CPU run; investigate first.")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = "mps"

    print("Loading clean_v3 dataset and matching 80 original baseline rows...", flush=True)
    validation = get_rows("validation")
    eval_rows = random.Random(SEED).sample(validation, SAMPLE_SIZE)
    prior_predictions = compare_prior(eval_rows)
    references = [r["target"] for r in eval_rows]
    original_score = float(sacrebleu.corpus_chrf(prior_predictions, [references]).score)
    print(f"Original reverse baseline on SAME 80 sentences: {original_score:.2f} chrF", flush=True)

    train = get_rows("train")
    random.Random(SEED + 17).shuffle(train)
    train = train[:TRAIN_UNIQUE]
    if len(train) < TRAIN_UNIQUE:
        raise RuntimeError("Training corpus is unexpectedly small")
    print(f"Unique training pairs: {len(train)}; epochs: {EPOCHS}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE, src_lang="arb_Arab", tgt_lang="tzm_Tfng")
    target_id = tokenizer.convert_tokens_to_ids("tzm_Tfng")
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise ValueError("NLLB missing target language token tzm_Tfng")
    print("Loading NLLB base and creating NEW reverse-direction adapter...", flush=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )).to(device)
    model.print_trainable_parameters()

    def translate(rows):
        model.eval()
        model.config.use_cache = True
        outputs = []
        for start in range(0, len(rows), 4):
            batch = rows[start : start + 4]
            x = tokenizer([r["source"] for r in batch], return_tensors="pt",
                          padding=True, truncation=True, max_length=MAX_INPUT).to(device)
            with torch.inference_mode():
                y = model.generate(**x, forced_bos_token_id=target_id,
                                   num_beams=3, max_new_tokens=80,
                                   repetition_penalty=1.12, no_repeat_ngram_size=3)
            outputs.extend(tokenizer.batch_decode(y, skip_special_tokens=True))
        return outputs

    def score(outputs):
        return float(sacrebleu.corpus_chrf(outputs, [references]).score)

    def save_adapter(path):
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    # Evaluate using the same decoding settings used in the old baseline as
    # another controlled point of comparison; only the LoRA comparisons below
    # use repetition guards.
    print("Evaluating new adapter before training...", flush=True)
    initial_preds = translate(eval_rows)
    initial_score = score(initial_preds)
    print(f"Initial adapter chrF (with repetition guard): {initial_score:.2f}", flush=True)
    print("Note: old zero-shot decoding did NOT use a repetition guard.", flush=True)

    best_score = initial_score
    best_step = 0
    best_preds = initial_preds
    save_adapter(OUT / "best_adapter")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    steps = 0
    presentations = 0
    started = time.monotonic()
    trace = [{"step": 0, "chrf": initial_score}]
    interrupted = False
    batch_size = MICROBATCH * ACCUM
    all_train = train * EPOCHS  # exactly two passes; keep the same train/validation split
    max_steps = min(MAX_STEPS, len(all_train) // batch_size)
    try:
        for step in range(1, max_steps + 1):
            if time.monotonic() - started >= MAX_TRAIN_SECONDS:
                print("Training time limit reached", flush=True)
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro in range(ACCUM):
                offset = ((step - 1) * ACCUM + micro) * MICROBATCH
                rows = all_train[offset : offset + MICROBATCH]
                x = tokenizer([r["source"] for r in rows], return_tensors="pt",
                              padding=True, truncation=True, max_length=MAX_INPUT).to(device)
                y = tokenizer(text_target=[r["target"] for r in rows],
                              return_tensors="pt", padding=True, truncation=True,
                              max_length=MAX_TARGET)["input_ids"].to(device)
                y[y == tokenizer.pad_token_id] = -100
                loss = model(**x, labels=y).loss
                (loss / ACCUM).backward()
                loss_value += float(loss.detach().item()) / ACCUM
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            steps = step
            presentations += batch_size
            if step % 25 == 0:
                print(f"step={step}/{max_steps} loss={loss_value:.3f} "
                      f"presentations={presentations} "
                      f"minutes={(time.monotonic()-started)/60:.1f}", flush=True)
            if step % CHECK_EVERY == 0:
                preds = translate(eval_rows)
                curr = score(preds)
                trace.append({"step": step, "chrf": curr})
                print(f"EVAL step={step} chrF={curr:.2f}", flush=True)
                if curr > best_score:
                    best_score, best_step, best_preds = curr, step, preds
                    save_adapter(OUT / "best_adapter")
                    print(f"Saved new best at step {best_step}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted: preserving current and best adapters", flush=True)

    save_adapter(OUT / "last_adapter")
    if steps and trace[-1]["step"] != steps:
        final_preds = translate(eval_rows)
        final_score = score(final_preds)
        trace.append({"step": steps, "chrf": final_score})
        if final_score > best_score:
            best_score, best_step, best_preds = final_score, steps, final_preds
            save_adapter(OUT / "best_adapter")
    else:
        final_score = trace[-1]["chrf"]

    with (OUT / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["arabic_input", "reference_tifinagh", "zero_shot_prior",
                         "initial_with_guard", "trained_best"])
        for r, prior, initial, best in zip(eval_rows, prior_predictions, initial_preds, best_preds):
            writer.writerow([r["source"], r["target"], prior, initial, best])
    results = {
        "distinct_train_pairs": len(train), "steps": steps,
        "training_presentations": presentations, "evaluation_rows": len(eval_rows),
        "original_baseline_chrf_no_guard": original_score,
        "initial_chrf_with_guard": initial_score,
        "best_chrf_with_guard": best_score,
        "last_chrf_with_guard": final_score, "best_step": best_step,
        "improvement_vs_guarded_baseline": best_score - initial_score,
        "evaluation_trace": trace, "interrupted": interrupted,
        "notes": "Target token tzm_Tfng is Central Atlas proxy, not Souss-specific."
    }
    write_json(OUT / "results.json", results)
    print("\n====== REVERSE LORA RESULTS ======", flush=True)
    for key, value in results.items():
        print(f"{key}: {value}", flush=True)
    print("\nFIRST EIGHT COMPARISONS", flush=True)
    for i, (row, before, after) in enumerate(zip(eval_rows, initial_preds, best_preds), 1):
        if i > 8:
            break
        print(f"\n{i}. AR:     {row['source']}")
        print(f"   BEFORE: {before}")
        print(f"   BEST:   {after}")
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()

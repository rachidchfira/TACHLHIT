#!/usr/bin/env python3
"""SoussVoice Stage 4: bounded NLLB LoRA continuation from Stage 3.

Training-only TAWNZA clean_v3; reproducible held-out validation; periodic
checkpointing. Never touches the old adapters or original datasets.
"""
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/external/tawnza/clean_v3'
STAGE3 = ROOT / 'experiments/tawnza/stage3/best_adapter'
OUT = ROOT / 'experiments/tawnza/stage4'
BASE = 'facebook/nllb-200-distilled-600M'
SEED = 20261008
# Bounded continuation: ~12,000 distinct new examples if all 1,500 steps finish.
MAX_STEPS = 1500
MAX_TRAIN_SECONDS = 60 * 60
MICROBATCH = 2
ACCUM_STEPS = 4
CHECK_EVERY = 250
VAL_SIZE = 240
MAX_SOURCE_TOKENS = 112
MAX_TARGET_TOKENS = 112
LR = 5e-5


def verify_paths():
    if not (STAGE3 / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Stage 3 adapter missing: {STAGE3}')
    for name in ('train', 'validation', 'test'):
        if not (DATA / f'{name}.parquet').is_file():
            raise FileNotFoundError(f'Missing dataset: {DATA / (name + ".parquet")}')
    if (OUT / 'results.json').exists():
        raise RuntimeError(f'{OUT}/results.json already exists. Do not overwrite a completed Stage 4 run.')
    OUT.mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    verify_paths()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = random.Random(SEED)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    print('Loading TAWNZA clean_v3 ...', flush=True)
    train = pq.read_table(DATA / 'train.parquet', columns=['source', 'target']).to_pylist()
    valid = pq.read_table(DATA / 'validation.parquet', columns=['source', 'target']).to_pylist()
    rng.shuffle(train)
    eval_rows = random.Random(SEED + 1).sample(valid, min(VAL_SIZE, len(valid)))
    # Fixed slice of 12,000 distinct rows; no duplicates due to replacement.
    train = train[:MAX_STEPS * MICROBATCH * ACCUM_STEPS]
    print(f'Train rows in this run: {len(train):,}; validation rows: {len(eval_rows)}', flush=True)
    save_json(OUT / 'configuration.json', {
        'source_dataset': str(DATA), 'starting_adapter': str(STAGE3),
        'seed': SEED, 'max_steps': MAX_STEPS, 'microbatch': MICROBATCH,
        'accumulation': ACCUM_STEPS, 'time_limit_seconds': MAX_TRAIN_SECONDS,
        'validation_count': len(eval_rows), 'learning_rate': LR,
        'date_utc': datetime.now(timezone.utc).isoformat(),
    })
    with (OUT / 'validation_sample.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['target', 'source'])
        writer.writeheader()
        writer.writerows(eval_rows)

    tokenizer = AutoTokenizer.from_pretrained(BASE, src_lang='tzm_Tfng', tgt_lang='arb_Arab')
    arabic_bos = tokenizer.convert_tokens_to_ids('arb_Arab')
    if arabic_bos == tokenizer.unk_token_id:
        raise ValueError('Arabic target language token is not supported by the tokenizer')
    print('Loading Stage 3 adapter...', flush=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base_model, str(STAGE3), is_trainable=True).to(device)
    model.print_trainable_parameters()

    def translate(data):
        model.eval()
        model.config.use_cache = True
        preds = []
        for start in range(0, len(data), 4):
            batch = data[start:start + 4]
            encoded = tokenizer(
                [x['target'] for x in batch], return_tensors='pt', padding=True,
                truncation=True, max_length=MAX_SOURCE_TOKENS,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, forced_bos_token_id=arabic_bos,
                    num_beams=3, max_new_tokens=96,
                )
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return preds

    def evaluate(step):
        preds = translate(eval_rows)
        refs = [r['source'] for r in eval_rows]
        score = float(sacrebleu.corpus_chrf(preds, [refs]).score)
        print(f'EVAL step={step}, chrF={score:.2f}', flush=True)
        return score, preds

    best_score, baseline_predictions = evaluate(0)
    baseline_score = best_score
    best_step = 0
    best_predictions = baseline_predictions
    # Best checkpoint is valid even when none of the additional steps help.
    model.save_pretrained(OUT / 'best_adapter')
    tokenizer.save_pretrained(OUT / 'best_adapter')

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    steps_completed = 0
    examples_presented = 0
    started = time.monotonic()
    trace = [{'step': 0, 'chrf': baseline_score}]
    interrupted = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if time.monotonic() - started >= MAX_TRAIN_SECONDS:
                print('Training time budget reached.', flush=True)
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro in range(ACCUM_STEPS):
                offset = ((step - 1) * ACCUM_STEPS + micro) * MICROBATCH
                batch = train[offset:offset + MICROBATCH]
                if not batch:
                    raise RuntimeError('Unexpected empty training batch')
                encoded = tokenizer(
                    [r['target'] for r in batch], return_tensors='pt',
                    padding=True, truncation=True, max_length=MAX_SOURCE_TOKENS,
                ).to(device)
                labels = tokenizer(
                    text_target=[r['source'] for r in batch],
                    return_tensors='pt', padding=True, truncation=True,
                    max_length=MAX_TARGET_TOKENS,
                )['input_ids'].to(device)
                labels[labels == tokenizer.pad_token_id] = -100
                output = model(**encoded, labels=labels)
                (output.loss / ACCUM_STEPS).backward()
                total_loss += float(output.loss.detach().item()) / ACCUM_STEPS
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            steps_completed = step
            examples_presented += MICROBATCH * ACCUM_STEPS
            if step % 25 == 0:
                print(f'Step {step}/{MAX_STEPS} | loss {total_loss:.3f} | '
                      f'examples {examples_presented:,} | '
                      f'train elapsed {(time.monotonic()-started)/60:.1f} min', flush=True)
            if step % CHECK_EVERY == 0:
                score, predictions = evaluate(step)
                trace.append({'step': step, 'chrf': score})
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_predictions = predictions
                    model.save_pretrained(OUT / 'best_adapter')
                    print(f'New best adapter saved: step {best_step}, chrF {best_score:.2f}', flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\nInterrupted by user: saving current adapter.', flush=True)

    # Save final adapter even if current performance regresses.
    model.save_pretrained(OUT / 'last_adapter')
    tokenizer.save_pretrained(OUT / 'last_adapter')
    if steps_completed and (not trace or trace[-1]['step'] != steps_completed):
        last_score, last_predictions = evaluate(steps_completed)
        trace.append({'step': steps_completed, 'chrf': last_score})
        if last_score > best_score:
            best_score = last_score
            best_step = steps_completed
            best_predictions = last_predictions
            model.save_pretrained(OUT / 'best_adapter')
    else:
        last_score = trace[-1]['chrf']
    with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tifinagh', 'arabic_reference', 'stage3_before', 'stage4_best'])
        for r, before, after in zip(eval_rows, baseline_predictions, best_predictions):
            writer.writerow([r['target'], r['source'], before, after])
    results = {
        'steps': steps_completed, 'examples_presented': examples_presented,
        'baseline_chrf': baseline_score, 'best_chrf': best_score,
        'last_chrf': last_score, 'best_step': best_step,
        'delta': best_score - baseline_score, 'interrupted': interrupted,
        'eval_trace': trace,
    }
    save_json(OUT / 'results.json', results)
    print('\n========== STAGE 4 RESULTS ==========', flush=True)
    for key, value in results.items():
        if key != 'eval_trace':
            print(f'{key}: {value}', flush=True)
    print('Evaluation trace:', trace, flush=True)
    print(f'Comparison: {OUT / "comparison.csv"}', flush=True)
    for i, (r, before, after) in enumerate(zip(eval_rows, baseline_predictions, best_predictions)):
        if i >= 6:
            break
        print(f'\nEXAMPLE {i+1} | REF: {r["source"]}\n  BEFORE: {before}\n  BEST:   {after}', flush=True)


if __name__ == '__main__':
    main()

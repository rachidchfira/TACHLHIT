#!/usr/bin/env python3
"""SoussVoice Stage 6: continue Arabic -> Tifinagh NLLB LoRA on new training pairs.

Start from reverse_lora_v1/best_adapter. Preserve all earlier adapters.
Use 240 DIFFERENT validation rows from the original 80-row reverse pilot,
not test rows. Output source + reference + before/best pairs for inspection.

Run inside ~/soussvoice:
  caffeinate -i python experiments/tawnza/train_soussvoice_reverse_stage6.py
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
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/external/tawnza/clean_v3'
PREVIOUS = ROOT / 'experiments/tawnza/reverse_lora_v1/best_adapter'
OUTPUT = ROOT / 'experiments/tawnza/reverse_stage6'
BASE = 'facebook/nllb-200-distilled-600M'

# These seeds reproduce the previous reverse pilot's selection.
PREVIOUS_SEED = 20261009
SEED = 20261010
DEV_SIZE = 240
NEW_TRAIN_PAIRS = 12000
BATCH = 2
ACCUM = 4
MAX_OPTIMIZER_STEPS = 1500
MAX_TRAIN_SECONDS = 45 * 60
CHECK_EVERY = 200
EARLY_STOP_PATIENCE = 3
MIN_GAIN = 0.15
LR = 7e-5
MAX_INPUT = 112
MAX_TARGET = 112


def read_rows(part):
    return pq.read_table(DATA / f'{part}.parquet', columns=['source', 'target']).to_pylist()


def dump_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    if not (PREVIOUS / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Missing previous reverse adapter: {PREVIOUS}')
    if (OUTPUT / 'results.json').exists():
        raise RuntimeError(f'Existing completed experiment at {OUTPUT}; refusing to overwrite')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; investigate before running large experiment on CPU')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = 'mps'

    print('Preparing evaluation independent of the original 80-sentence reverse pilot...', flush=True)
    valid_rows = read_rows('validation')
    original_80 = random.Random(PREVIOUS_SEED).sample(valid_rows, 80)
    original_keys = {(r['source'], r['target']) for r in original_80}
    valid_candidates = [r for r in valid_rows if (r['source'], r['target']) not in original_keys]
    dev_rows = random.Random(SEED).sample(valid_candidates, DEV_SIZE)
    assert not original_keys.intersection((r['source'], r['target']) for r in dev_rows)
    references = [r['target'] for r in dev_rows]

    train_rows = read_rows('train')
    # Previous reverse pilot selected 4,000 after this exact shuffle.
    random.Random(PREVIOUS_SEED + 17).shuffle(train_rows)
    already_sampled = train_rows[:4000]
    prior_keys = {(r['source'], r['target']) for r in already_sampled}
    candidates = [r for r in train_rows[4000:] if (r['source'], r['target']) not in prior_keys]
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:NEW_TRAIN_PAIRS]
    if len(selected) != NEW_TRAIN_PAIRS:
        raise RuntimeError(f'Need {NEW_TRAIN_PAIRS} new pairs; found {len(selected)}')
    assert not prior_keys.intersection((r['source'], r['target']) for r in selected)
    # Deliberately a single pass over fresh examples; choose repetition later from evidence.
    print(f'Additional distinct training pairs available this run: {len(selected):,}', flush=True)
    print(f'New validation examples: {len(dev_rows)} (original reverse 80 excluded)', flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, src_lang='arb_Arab', tgt_lang='tzm_Tfng')
    target_id = tok.convert_tokens_to_ids('tzm_Tfng')
    assert target_id != tok.unk_token_id
    print('Loading NLLB + trained reverse adapter...', flush=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base, str(PREVIOUS), is_trainable=True).to(device)
    model.print_trainable_parameters()

    def predict(rows):
        model.eval()
        model.config.use_cache = True
        output = []
        for i in range(0, len(rows), 4):
            mini = rows[i:i+4]
            tokens = tok([r['source'] for r in mini], padding=True, truncation=True,
                         max_length=MAX_INPUT, return_tensors='pt').to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens, forced_bos_token_id=target_id, num_beams=3,
                    max_new_tokens=80, repetition_penalty=1.12, no_repeat_ngram_size=3
                )
            output.extend(tok.batch_decode(generated, skip_special_tokens=True))
        return output

    def score(outputs):
        return float(sacrebleu.corpus_chrf(outputs, [references]).score)

    def save_adapter(folder):
        model.save_pretrained(folder)
        tok.save_pretrained(folder)

    print('Initial validation on 240 fresh sentences...', flush=True)
    original_preds = predict(dev_rows)
    original_score = score(original_preds)
    print(f'Starting chrF: {original_score:.2f}', flush=True)
    best_score, best_step, best_preds = original_score, 0, original_preds
    save_adapter(OUTPUT / 'best_adapter')

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    steps, presented = 0, 0
    trace = [{'step': 0, 'chrf': original_score}]
    no_gain_checks = 0
    started = time.monotonic()
    stopped_for = 'completed'
    max_steps = min(MAX_OPTIMIZER_STEPS, len(selected) // (BATCH * ACCUM))

    try:
        for step in range(1, max_steps + 1):
            if time.monotonic() - started >= MAX_TRAIN_SECONDS:
                stopped_for = 'training_time_limit'
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for micro in range(ACCUM):
                offset = ((step - 1) * ACCUM + micro) * BATCH
                rows = selected[offset:offset+BATCH]
                x = tok([r['source'] for r in rows], padding=True, truncation=True,
                        max_length=MAX_INPUT, return_tensors='pt').to(device)
                y = tok(text_target=[r['target'] for r in rows], padding=True, truncation=True,
                        max_length=MAX_TARGET, return_tensors='pt')['input_ids'].to(device)
                y[y == tok.pad_token_id] = -100
                loss = model(**x, labels=y).loss
                (loss / ACCUM).backward()
                loss_total += float(loss.detach().item()) / ACCUM
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            steps = step
            presented += BATCH * ACCUM
            if step % 25 == 0:
                print(f'step={step}/{max_steps} loss={loss_total:.3f} '
                      f'examples={presented} elapsed={(time.monotonic()-started)/60:.1f}m', flush=True)
            if step % CHECK_EVERY == 0:
                latest_preds = predict(dev_rows)
                latest = score(latest_preds)
                trace.append({'step': step, 'chrf': latest})
                print(f'EVAL step={step} chrF={latest:.2f} best={best_score:.2f}', flush=True)
                # Save every observed higher-scoring checkpoint; patience ignores trivial gains.
                gain = latest - best_score
                if gain > 0:
                    best_score, best_step, best_preds = latest, step, latest_preds
                    save_adapter(OUTPUT / 'best_adapter')
                if gain >= MIN_GAIN:
                    no_gain_checks = 0
                else:
                    no_gain_checks += 1
                if no_gain_checks >= EARLY_STOP_PATIENCE:
                    stopped_for = 'early_stopping'
                    print('Early stopping: no meaningful gains at three evaluations.', flush=True)
                    break
    except KeyboardInterrupt:
        stopped_for = 'keyboard_interrupt'
        print('Interrupted; saving the latest and best adapters.', flush=True)

    save_adapter(OUTPUT / 'last_adapter')
    if steps and trace[-1]['step'] != steps:
        latest_preds = predict(dev_rows)
        latest = score(latest_preds)
        trace.append({'step': steps, 'chrf': latest})
        if latest > best_score:
            best_score, best_step, best_preds = latest, steps, latest_preds
            save_adapter(OUTPUT / 'best_adapter')
    last_score = trace[-1]['chrf']

    with (OUTPUT / 'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arabic_input', 'reference_tifinagh', 'before', 'best'])
        for r, before, best in zip(dev_rows, original_preds, best_preds):
            writer.writerow([r['source'], r['target'], before, best])
    results = {
        'step_count': steps, 'sentence_presentations': presented,
        'distinct_new_training_pairs_selected': len(selected),
        'fresh_validation_size': len(dev_rows),
        'initial_chrf': original_score, 'best_chrf': best_score,
        'last_chrf': last_score, 'best_step': best_step,
        'gain_on_new_validation': best_score - original_score,
        'evaluations': trace, 'stop_reason': stopped_for,
        'note': 'tzm_Tfng is a Central Atlas proxy; scores do not verify Souss naturalness.'
    }
    dump_json(OUTPUT / 'results.json', results)
    print('\n======== REVERSE STAGE 6 RESULTS ========', flush=True)
    for k, v in results.items():
        print(f'{k}: {v}', flush=True)
    print('\nFIRST EIGHT EXAMPLES', flush=True)
    for i, (r, old, new) in enumerate(zip(dev_rows, original_preds, best_preds), 1):
        if i > 8:
            break
        print(f'\n{i}. AR: {r["source"]}')
        print(f'   REF: {r["target"]}')
        print(f'   BEFORE: {old}')
        print(f'   BEST: {new}')
    print(f'\nSaved: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()

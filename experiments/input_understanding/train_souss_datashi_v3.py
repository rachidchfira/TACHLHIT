#!/usr/bin/env python3
"""SoussVoice V3: ONE bounded DATASHI domain-adaptation experiment.

Source: datasets/understanding_v3/datashi/train.csv (new DATASHI only)
Selection: datasets/understanding_v3/datashi/validation.csv (not test)
Never train on native gold, original validation/test or DATASHI benchmark/test.
Warm-start: best conversational-weighted V2 LoRA (not modified).

Usage from ~/soussvoice, with the existing venv active:
 python experiments/input_understanding/train_souss_datashi_v3.py --self-test
 python experiments/input_understanding/train_souss_datashi_v3.py --preflight-only
 caffeinate -i python -u experiments/input_understanding/train_souss_datashi_v3.py
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAT = ROOT / 'datasets/understanding_v3/datashi'
OLD = ROOT / 'datasets/reverse_translation/v2'
BENCH = ROOT / 'experiments/input_understanding/datashi_comparison_v1/comparisons.csv'
GOLD = ROOT / 'datasets/native_gold_v1/native_verified.csv'
START = ROOT / 'experiments/input_understanding/multilingual_lora_weighted_v2/best_adapter'
BASE = 'vonjack/opus-mt-mul-en-big'
OUT = ROOT / 'experiments/input_understanding/datashi_adaptation_v3'
SEED = 20261009


def key(s):
    # Stronger than whitespace-normalized matching: punctuation cannot hide overlap.
    s = unicodedata.normalize('NFC', str(s)).casefold()
    return re.sub(r'[\W_]+', '', s, flags=re.UNICODE)


def read(path, source='shi_latn', target='en'):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        if not {source, target}.issubset(r.fieldnames or []):
            raise ValueError(f'Missing columns {source}/{target} in {path}')
        return [{'shi': row[source].strip(), 'en': row[target].strip()}
                for row in r if row[source].strip() and row[target].strip()]


def keys(path, col):
    if not path.exists():
        return set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        if col not in (r.fieldnames or []):
            raise ValueError(f'Missing {col}: {path}')
        return {key(row[col]) for row in r if row[col].strip()}


def hashfile(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_data():
    train = read(DAT / 'train.csv')
    val = read(DAT / 'validation.csv')
    test = read(DAT / 'test.csv')  # Leakage audit ONLY. Never translate or train on it.
    protected = (keys(BENCH, 'shi') | keys(GOLD, 'shi_latn') |
                 keys(OLD / 'validation.csv', 'shi_latn') |
                 keys(OLD / 'test.csv', 'shi_latn'))
    dat_test = {key(r['shi']) for r in test}
    val_keys = {key(r['shi']) for r in val}
    # Model V2 was already trained on original train: remove any such overlaps
    # from DATASHI validation to retain some independent validation signal.
    old_train = keys(OLD / 'train.csv', 'shi_latn')
    bench_keys = keys(BENCH, 'shi')
    bad_val = val_keys & (old_train | protected | dat_test)
    if bad_val:
        raise RuntimeError(f'DATASHI validation has {len(bad_val)} prior-data/test overlaps; '
                           'do not silently evaluate contaminated validation inputs.')
    bad_test = dat_test & (old_train | protected | val_keys)
    if bad_test:
        raise RuntimeError(f'DATASHI test has {len(bad_test)} prior-data/eval overlaps; '
                           'fix dataset splitting before proceeding.')
    held = protected | val_keys | dat_test
    clean = []
    seen = set()
    removed = {'held_out_overlap': 0, 'duplicate': 0}
    for row in train:
        k = key(row['shi'])
        if not k or not row['en']:
            continue
        if k in held:
            removed['held_out_overlap'] += 1
        elif k in seen:
            removed['duplicate'] += 1
        else:
            clean.append(row)
            seen.add(k)
    if len(clean) < 800 or len(val) < 100:
        raise RuntimeError(f'Too few safe data: train={len(clean)}, val={len(val)}')
    if len(bench_keys) < 90:
        raise RuntimeError('External 100-item DATASHI benchmark is incomplete.')
    return clean, val, len(test), removed


def self_test():
    assert key(' Awi t! ') == key('awi   t.')
    assert key('Iṛmi sul.') == key(' IṚMI SUL!')
    assert key('irɣa sul ljaww') != key('Iṛmi sul.')
    print('SELF-TEST PASSED: normalized overlap and warm/tired contrast.')
    print('No datasets, checkpoints or models modified.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=40)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--max-minutes', type=float, default=35)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.steps <= 0 or args.batch_size <= 0 or args.accum <= 0 or args.eval_every <= 0 or args.lr <= 0:
        p.error('Training parameters must be positive.')

    train, val, n_test, removed = safe_data()
    print(f'SAFE DATA: train={len(train)}, validation={len(val)}, '
          f'reserved DATASHI test={n_test}; excluded train={removed}', flush=True)
    if args.steps * args.batch_size * args.accum > len(train):
        raise RuntimeError('More than one training pass requested. Reduce --steps.')
    if not (START / 'adapter_config.json').exists():
        raise FileNotFoundError(f'V2 best adapter not found: {START}')
    if OUT.exists():
        raise RuntimeError(f'Output directory already exists: {OUT}\n'
                           'Refusing overwrite. Preserve the first controlled experiment.')
    print('START: V2 best adapter; base:', BASE, flush=True)
    print('DATASHI benchmark/native gold: protected. DATASHI test: audit only.', flush=True)
    print('Source dataset SHA256:', hashfile(DAT / 'train.csv'), flush=True)
    if args.preflight_only:
        print('PREFLIGHT PASSED. No training or files written.')
        return

    import sacrebleu
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS unavailable; refusing long CPU training.')
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = 'mps'
    random.Random(SEED).shuffle(train)
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE).to(device, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(START), is_trainable=True).to(device)
    model.print_trainable_parameters()
    trainables = [param for param in model.parameters() if param.requires_grad]
    if not trainables:
        raise RuntimeError('No trainable LoRA parameters')
    optimizer = torch.optim.AdamW(trainables, lr=args.lr)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'manifest.json').write_text(json.dumps({
        'base': BASE, 'start_adapter': str(START), 'training_file_sha256': hashfile(DAT / 'train.csv'),
        'training_examples': len(train), 'validation_examples': len(val), 'test_examples_untouched': n_test,
        'removed': removed, 'steps': args.steps, 'batch_size': args.batch_size,
        'accum': args.accum, 'lr': args.lr, 'seed': SEED,
    }, indent=2), encoding='utf-8')

    def translate(rows):
        model.eval()
        model.config.use_cache = True
        results = []
        for i in range(0, len(rows), 4):
            inp = tok([r['shi'] for r in rows[i:i+4]], padding=True,
                      truncation=True, max_length=112, return_tensors='pt').to(device)
            with torch.inference_mode():
                output = model.generate(**inp, num_beams=3, max_length=96)
            results.extend(tok.batch_decode(output, skip_special_tokens=True))
        return results

    def evaluate(step, name):
        pred = translate(val)
        chrf = float(sacrebleu.corpus_chrf(pred, [[r['en'] for r in val]]).score)
        path = OUT / f'validation_{name}.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['shi', 'reference_en', 'prediction_en'])
            for row, prediction in zip(val, pred):
                writer.writerow([row['shi'], row['en'], prediction])
        print(f'VALIDATION step {step}/{args.steps} chrF={chrf:.2f}', flush=True)
        return chrf

    baseline = evaluate(0, 'baseline')
    best = baseline
    best_step = 0
    trace = [{'step': 0, 'chrf': baseline}]
    # Save warm-start as step-0 best: never accidentally replace it with a worse V3.
    model.save_pretrained(OUT / 'best_adapter')
    tok.save_pretrained(OUT / 'best_adapter')
    start_time = time.monotonic()
    completed = 0
    try:
        for step in range(1, args.steps + 1):
            if time.monotonic() - start_time > args.max_minutes * 60:
                print('Time budget reached; ending at last completed step.', flush=True)
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro in range(args.accum):
                begin = ((step - 1) * args.accum + micro) * args.batch_size
                rows = train[begin:begin + args.batch_size]
                inp = tok([r['shi'] for r in rows], padding=True,
                          truncation=True, max_length=112, return_tensors='pt').to(device)
                labels = tok(text_target=[r['en'] for r in rows], padding=True,
                             truncation=True, max_length=112, return_tensors='pt')['input_ids'].to(device)
                labels[labels == tok.pad_token_id] = -100
                loss = model(**inp, labels=labels).loss
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f'Nonfinite loss step {step}')
                (loss / args.accum).backward()
                total_loss += float(loss.detach()) / args.accum
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            optimizer.step()
            completed = step
            if step % 10 == 0 or step == 1:
                print(f'STEP {step}/{args.steps} loss={total_loss:.3f} '
                      f'min={(time.monotonic()-start_time)/60:.1f}', flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                current = evaluate(step, f'step_{step}')
                trace.append({'step': step, 'chrf': current})
                if current > best:
                    best, best_step = current, step
                    model.save_pretrained(OUT / 'best_adapter')
                    tok.save_pretrained(OUT / 'best_adapter')
                    print('NEW VALIDATION BEST', step, flush=True)
                (OUT / 'results.json').write_text(json.dumps({
                    'baseline_chrf': baseline, 'best_chrf': best, 'best_step': best_step,
                    'completed_steps': completed, 'trace': trace, 'test_evaluated': False
                }, indent=2), encoding='utf-8')
    except KeyboardInterrupt:
        print('Interrupted. Previously saved BEST checkpoint is preserved.', flush=True)
    print('\n=== DATASHI V3 CONTROLLED EXPERIMENT ===', flush=True)
    print('Baseline validation chrF:', round(baseline, 2), flush=True)
    print('Best validation chrF:', round(best, 2), 'step:', best_step, flush=True)
    print('Completed steps:', completed, '/', args.steps, flush=True)
    print('Best adapter:', OUT / 'best_adapter', flush=True)
    print('Validation predictions:', OUT, flush=True)
    print('No external benchmark/native gold/test inference was performed.', flush=True)
    print('Do NOT deploy from chrF alone: compare semantic predictions first.', flush=True)

if __name__ == '__main__':
    main()

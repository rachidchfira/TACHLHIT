#!/usr/bin/env python3
"""SoussVoice Input V2: controlled, conversation-weighted LoRA continuation.

Training source: datasets/understanding_v2/train_weighted.csv.
Validation/test: untouched datasets/reverse_translation/v2/{validation,test}.csv.
Warm-starts from full V1's BEST adapter (step 2100), NOT the final adapter.

Usage (project root, venv activated):
  python experiments/input_understanding/train_souss_input_v2.py --self-test
  python experiments/input_understanding/train_souss_input_v2.py --preflight-only --steps 600 --val-samples 200
  caffeinate -i python -u experiments/input_understanding/train_souss_input_v2.py --steps 600 --val-samples 200 --eval-every 300 --max-minutes 30
  # Resume ONLY using identical parameters and --resume.

Weighted duplicate source/target pairs are preserved intentionally, but all
exact and punctuation-normalized overlaps with held-out inputs are removed.
Never modifies V1 outputs or datasets. Uses a separate V2 experiment folder.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import csv
import json
import random
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/reverse_translation/v2'
TRAIN_FILE = ROOT / 'datasets/understanding_v2/train_weighted.csv'
START_ADAPTER = ROOT / 'experiments/input_understanding/multilingual_lora_full_v1/best_adapter'
OUT = ROOT / 'experiments/input_understanding/multilingual_lora_weighted_v2'
BASE = 'vonjack/opus-mt-mul-en-big'
SEED = 20261009
PROBES = [
    ('Ur riɣ ad skrɣ imukrisn.', "I don't want to cause problems."),
    ('Mani s rad tftut azkka?', 'Where will you go tomorrow?'),
    ('Riɣ ad suɣ aman.', 'I want to drink water.'),
]


def normalized(s):
    return ' '.join(unicodedata.normalize('NFC', s).casefold().split())


def read_split(name):
    path = TRAIN_FILE if name == 'train' else DATA / f'{name}.csv'
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'shi_latn', 'en'}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f'Missing {required} columns: {path}')
        rows = [{'source': r['shi_latn'].strip(), 'target': r['en'].strip()}
                for r in reader if r['shi_latn'].strip() and r['en'].strip()]
    if not rows:
        raise RuntimeError(f'No usable rows in {path}')
    return rows


def punctuation_key(s):
    # Match the conservative leakage filter used in build_souss_dataset_v1.py.
    return re.sub(r'[\s\W_]+', '', normalized(s), flags=re.UNICODE)


def safe_training_rows(train, validation, test):
    """Keep intentional weighting duplicates, but exclude held-out inputs.

    The original V1 function removed duplicate source+English pairs, which
    would silently destroy our conversation oversampling. Do NOT reuse it.
    """
    held = validation + test + [{'source': s} for s, _ in PROBES]
    held_exact = {normalized(r['source']) for r in held}
    held_punct = {punctuation_key(r['source']) for r in held}
    result, removed = [], 0
    for row in train:
        if (normalized(row['source']) in held_exact or
                punctuation_key(row['source']) in held_punct):
            removed += 1
            continue
        result.append(row)  # Important: preserve repeated training presentations
    return result, removed


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def self_test():
    data = [
        {'source': 'Dars imukrisn.', 'target': 'He has problems.'},
        {'source': 'Dars imukrisn.', 'target': 'He has problems.'},
        {'source': 'Awi t.', 'target': 'Take it.'},
        {'source': 'Is d kmmi aya?', 'target': 'Is it you?'},
    ]
    # Second sample is intentionally repeated; punctuation variant leaks.
    filtered, dropped = safe_training_rows(
        data, [{'source': 'Awi t!'}], [])
    assert dropped == 1 and len(filtered) == 3, (dropped, filtered)
    assert filtered[0] == filtered[1], 'Weighting duplicates were lost'
    assert normalized('  A\u0302  B ') == normalized('Â B')
    assert punctuation_key('Awi t!') == punctuation_key('Awi t.')
    assert len(PROBES) == 3
    assert OUT != START_ADAPTER.parent
    print('V2 SELF-TEST PASSED: repeat weighting retained; held-out punctuation/exact overlap blocked.')
    print('No models, files, or checkpoints modified.')


def run(args):
    # Import optional, heavier dependencies only for the actual experiment.
    import sacrebleu
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS not available. Refusing accidental long CPU training.')
    device = 'mps'
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    if not TRAIN_FILE.is_file():
        raise RuntimeError(f'Weighted training file missing: {TRAIN_FILE}')
    if not (START_ADAPTER / 'adapter_config.json').is_file():
        raise RuntimeError(f'V1 best adapter missing: {START_ADAPTER}')
    data_hash = sha256_file(TRAIN_FILE)
    train = read_split('train')
    validation = read_split('validation')
    test = read_split('test')
    safe_train, removed = safe_training_rows(train, validation, test)
    if len(safe_train) < 1000:
        raise RuntimeError('Unexpectedly few safe training rows; stop and inspect datasets.')
    if args.steps * args.microbatch * args.accum > len(safe_train):
        raise RuntimeError('V2 experiment exceeds one weighted pass; reduce --steps.')
    print(f'Weighted train {len(train)} -> {len(safe_train)} after held-out filtering '
          f'({removed} removed); validation {len(validation)}, untouched test {len(test)}.', flush=True)
    print(f'Warm-start adapter: {START_ADAPTER}', flush=True)
    print(f'Weighted dataset sha256: {data_hash}', flush=True)
    sample = random.Random(SEED).sample(validation, min(args.val_samples, len(validation)))
    shuffled = safe_train.copy()
    random.Random(SEED + 1).shuffle(shuffled)
    print('Example:', shuffled[0], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    state_file = OUT / 'last_optimizer_state.pt'
    last_dir = OUT / 'last_adapter'
    best_dir = OUT / 'best_adapter'
    run_file = OUT / 'run_state.json'
    if not args.resume and (state_file.exists() or run_file.exists()):
        raise RuntimeError(f'An experiment already exists at {OUT}. '
                           'Use --resume; original run will not be overwritten.')
    if args.resume and not (state_file.is_file() and last_dir.is_dir() and run_file.is_file()):
        raise RuntimeError('No resumable run exists. Run without --resume first.')

    print(f'Loading cached pretrained model {BASE}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    # FP32 training avoids half-precision optimizer instability on Apple MPS.
    base = base.to(device=device, dtype=torch.float32)
    if args.resume:
        model = PeftModel.from_pretrained(base, last_dir, is_trainable=True)
    else:
        model = PeftModel.from_pretrained(base, str(START_ADAPTER), is_trainable=True)
    model.to(device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0.0)
    start_step = 0
    trace = []
    if args.resume:
        prior = json.loads(run_file.read_text(encoding='utf-8'))
        if (prior['steps_planned'] != args.steps or
                prior['microbatch'] != args.microbatch or
                prior['accum'] != args.accum or
                prior['val_samples'] != args.val_samples or
                prior['lr'] != args.lr or
                prior.get('train_sha256') != data_hash or
                prior.get('eval_every') != args.eval_every):
            raise RuntimeError('Resume parameters differ from original experiment.')
        start_step = int(prior['last_step'])
        trace = prior['trace']
        optimizer.load_state_dict(torch.load(state_file, map_location=device,
                                             weights_only=True))
        baseline = prior['baseline']
        use_tag = bool(prior['use_tag'])
        best_score = float(prior['best_chrf'])
        best_step = int(prior['best_step'])
        print(f'Resumed at step {start_step}, best step {best_step}.', flush=True)
    else:
        baseline, use_tag, best_score, best_step = {}, False, -1.0, 0

    def source_text(s):
        return ('>>shi<< ' if use_tag else '') + s

    def translate(sentences, batch_size=4):
        model.eval()
        model.config.use_cache = True
        outputs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded = tokenizer([source_text(s) for s in batch], padding=True,
                                truncation=True, max_length=112, return_tensors='pt').to(device)
            with torch.inference_mode():
                generated = model.generate(**encoded, num_beams=3, max_length=96)
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs

    refs = [r['target'] for r in sample]

    def evaluate(step):
        predictions = translate([r['source'] for r in sample])
        score = float(sacrebleu.corpus_chrf(predictions, [refs]).score)
        print(f'VALIDATION step={step}: chrF={score:.2f}  (sample={len(sample)})', flush=True)
        return score, predictions

    def save_preds(filename, predictions):
        with (OUT / filename).open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['shi_latn', 'reference_en', 'prediction_en'])
            for row, pred in zip(sample, predictions):
                writer.writerow([row['source'], row['target'], pred])

    if not args.resume:
        # Compare no-tag vs literal OPUS language marker. Some Marian tokenizers
        # do not expose '>>shi<<' as a dedicated special token; test empirically.
        for tag in (False, True):
            use_tag = tag
            score, preds = evaluate(0)
            key = 'literal_shi_prefix' if tag else 'no_prefix'
            baseline[key] = score
            save_preds(f'baseline_{key}.csv', preds)
            print(f'BASELINE {key}: {score:.2f}', flush=True)
        use_tag = baseline['literal_shi_prefix'] > baseline['no_prefix']
        best_score = baseline['literal_shi_prefix' if use_tag else 'no_prefix']
        print(f'Selected source format: {"literal >>shi<< prefix" if use_tag else "no prefix"}', flush=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        trace = [{'step': 0, 'chrf': best_score}]
        best_step = 0

        print('\nNATIVE-VERIFIED PROBES (V1 warm-start baseline; kept out of V2 training):', flush=True)
        for (sentence, ref), pred in zip(PROBES, translate([x for x, _ in PROBES])):
            print(f'  SHI: {sentence}\n  REF: {ref}\n  PRED: {pred}', flush=True)
        if args.preflight_only:
            save_json(OUT / 'preflight.json', {'baseline': baseline,
                       'use_tag': use_tag, 'train_count': len(safe_train),
                       'validation_sample': len(sample), 'test_used': False})
            print('PREFLIGHT COMPLETE — no training occurred.', flush=True)
            return

    def store_progress(step):
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        # Store optimizer state for exact AdamW continuation from this step.
        torch.save(optimizer.state_dict(), state_file)
        save_json(run_file, {'steps_planned': args.steps, 'last_step': step,
                            'microbatch': args.microbatch, 'accum': args.accum,
                            'val_samples': args.val_samples, 'lr': args.lr,
                            'train_sha256': data_hash, 'eval_every': args.eval_every,
                            'starting_adapter': str(START_ADAPTER),
                            'baseline': baseline, 'use_tag': use_tag,
                            'best_chrf': best_score, 'best_step': best_step,
                            'trace': trace, 'test_used': False})

    last_completed = start_step
    beginning = time.monotonic()
    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        for step in range(start_step + 1, args.steps + 1):
            if time.monotonic() - beginning > args.max_minutes * 60:
                print('Training time budget reached; checkpointing.', flush=True)
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            loss_num = 0.0
            for micro in range(args.accum):
                offset = ((step - 1) * args.accum + micro) * args.microbatch
                rows = shuffled[offset:offset + args.microbatch]
                inputs = tokenizer([source_text(r['source']) for r in rows],
                                   padding=True, truncation=True, max_length=112,
                                   return_tensors='pt').to(device)
                targets = tokenizer(text_target=[r['target'] for r in rows],
                                    padding=True, truncation=True, max_length=112,
                                    return_tensors='pt')['input_ids'].to(device)
                targets[targets == tokenizer.pad_token_id] = -100
                loss = model(**inputs, labels=targets).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f'Nonfinite loss at step {step}; aborting.')
                (loss / args.accum).backward()
                loss_num += loss.detach().item() / args.accum
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            last_completed = step
            if step % 10 == 0 or step == 1:
                print(f'STEP {step}/{args.steps} loss={loss_num:.3f} '
                      f'min={(time.monotonic()-beginning)/60:.1f}', flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                curr, pred = evaluate(step)
                trace.append({'step': step, 'chrf': curr})
                if curr > best_score:
                    best_score, best_step = curr, step
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    save_preds('best_validation.csv', pred)
                    print('Saved IMPROVED adapter at step', step, flush=True)
                store_progress(step)
    except KeyboardInterrupt:
        print('Interrupted: saving adapter and optimizer state...', flush=True)
    finally:
        if last_completed > start_step and (not run_file.is_file() or
                int(json.loads(run_file.read_text())['last_step']) != last_completed):
            store_progress(last_completed)
        elif not run_file.exists():
            store_progress(last_completed)

    print('\n=== WEIGHTED V2 RESULT ===', flush=True)
    print('Baseline no prefix:', round(baseline['no_prefix'], 2), flush=True)
    print('Baseline >>shi<<:', round(baseline['literal_shi_prefix'], 2), flush=True)
    print('Best validation chrF:', round(best_score, 2), 'step:', best_step, flush=True)
    print('Completed steps:', last_completed, '/', args.steps, flush=True)
    print('Adapter:', best_dir, flush=True)
    print('UNTOUCHED test split: not evaluated; reserve for later.', flush=True)
    print('Evaluate meaning manually before accepting score-based improvements.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--val-samples', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=300)
    p.add_argument('--max-minutes', type=float, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.steps < 1 or args.microbatch < 1 or args.accum < 1 or args.val_samples < 2:
        p.error('Invalid steps/batch/validation sizes.')
    if args.resume and args.preflight_only:
        p.error('--resume and --preflight-only cannot be used together.')
    run(args)


if __name__ == '__main__':
    main()

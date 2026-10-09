#!/usr/bin/env python3
"""Compare existing SoussVoice input models on held-out DATASHI examples.

Runs inference ONLY. Never changes existing models/datasets/checkpoints.
"""
import argparse
import csv
import gc
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/external/datashi/inputs.csv'
SPLITS = ROOT / 'datasets/reverse_translation/v2'
OUT = ROOT / 'experiments/input_understanding/datashi_comparison_v1'
BASE = 'vonjack/opus-mt-mul-en-big'
MODELS = [
    ('marian_original', ROOT / 'experiments/language-adaptation/marian_runs/v1/best_model'),
    ('multilingual_v1', ROOT / 'experiments/input_understanding/multilingual_lora_full_v1/best_adapter'),
    ('multilingual_v2', ROOT / 'experiments/input_understanding/multilingual_lora_weighted_v2/best_adapter'),
]


def norm(text):
    text = unicodedata.normalize('NFC', (text or '')).casefold()
    return ' '.join(text.split())


def loose_norm(text):
    text = norm(text)
    text = ''.join(' ' if unicodedata.category(c).startswith(('P', 'S')) else c for c in text)
    return ' '.join(text.split())


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def build_selection(n):
    source = load_csv(DATA)
    forbidden_shi, forbidden_en = set(), set()
    for split in ('train', 'validation', 'test'):
        for row in load_csv(SPLITS / (split + '.csv')):
            forbidden_shi.add(loose_norm(row['shi_latn']))
            forbidden_en.add(loose_norm(row['en']))
    groups = defaultdict(list)
    removed, duplicates = 0, 0
    seen = set()
    for row in source:
        shi, en = (row.get('SHI-s') or '').strip(), (row.get('EN') or '').strip()
        if not shi or not en:
            removed += 1
            continue
        kshi, ken = loose_norm(shi), loose_norm(en)
        if kshi in forbidden_shi or ken in forbidden_en:
            removed += 1
            continue
        if (kshi, ken) in seen:
            duplicates += 1
            continue
        seen.add((kshi, ken))
        groups[(row.get('CATEGORY') or 'unknown').strip()].append({
            'category': (row.get('CATEGORY') or 'unknown').strip(),
            'shi': shi,
            'reference_en': en,
        })
    rng = random.Random(2026)
    for group in groups.values():
        rng.shuffle(group)
    cats = sorted(groups)
    rng.shuffle(cats)
    selection = []
    # Balanced round-robin by category; no model outputs influence sample selection.
    while len(selection) < n and any(groups.values()):
        for cat in cats:
            if groups[cat] and len(selection) < n:
                selection.append(groups[cat].pop())
    print(f'DATASHI: {len(source)} total, {removed} excluded for split overlap/empty, '
          f'{duplicates} duplicate pairs; {len(selection)} selected across {len(set(x["category"] for x in selection))} categories.')
    if len(selection) < n:
        print(f'Warning: only {len(selection)} eligible examples available.')
    return selection


def translate(model, tokenizer, rows, device, batch_size=4):
    import torch
    outputs = []
    for i in range(0, len(rows), batch_size):
        sources = [x['shi'] for x in rows[i:i + batch_size]]
        encoded = tokenizer(sources, padding=True, truncation=True, max_length=160, return_tensors='pt').to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=80, num_beams=4)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        if (i + batch_size) % 20 == 0 or i + batch_size >= len(rows):
            print(f'  translated {min(i + batch_size, len(rows))}/{len(rows)}', flush=True)
    return outputs


def run(n, device_option):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    import sacrebleu
    if OUT.exists():
        raise SystemExit(f'Output directory already exists: {OUT}\n'
                         'Preserved for reproducibility. Rename it before running again.')
    for label, path in MODELS:
        if not path.is_dir():
            raise SystemExit(f'Missing {label}: {path}')
    rows = build_selection(n)
    if not rows:
        raise SystemExit('No eligible examples to evaluate.')
    # Freeze the test inputs BEFORE any model inference.
    OUT.mkdir(parents=True)
    write_csv(OUT / 'selected.csv', rows, ['category', 'shi', 'reference_en'])
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device_option == 'auto' else device_option
    print(f'Inference device: {device}; saved fixed inputs to {OUT / "selected.csv"}')
    refs = [r['reference_en'] for r in rows]
    metrics = {}
    for label, path in MODELS:
        print(f'\n=== {label} ===', flush=True)
        if label == 'marian_original':
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(BASE)
            backbone = AutoModelForSeq2SeqLM.from_pretrained(BASE)
            model = PeftModel.from_pretrained(backbone, str(path))
        model.to(device).eval()
        predictions = translate(model, tokenizer, rows, device)
        for row, prediction in zip(rows, predictions):
            row[label] = prediction
        score = sacrebleu.corpus_chrf(predictions, [refs]).score
        metrics[label] = round(score, 2)
        print(f'{label}: chrF {score:.2f}', flush=True)
        write_csv(OUT / 'comparisons.csv', rows, ['category', 'shi', 'reference_en'] + list(metrics))
        (OUT / 'scores.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
        del model
        if 'backbone' in locals():
            del backbone
        del tokenizer
        gc.collect()
        if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    print('\n=== FINAL DATASHI chrF ===')
    for label, score in metrics.items():
        print(f'{label}: {score:.2f}')
    print(f'\nFiles: {OUT / "comparisons.csv"}')
    print('Interpret scores with semantic examples. No models were trained.')


def self_test():
    assert loose_norm('Ssurf-iyi...') == loose_norm('Ssurf iyi!')
    assert norm('ƔZZIFƔ.').startswith('ɣzzif')
    assert loose_norm('  Awi t. ') == 'awi t'
    assert loose_norm('') == ''
    print('SELF-TEST PASSED: normalization and safe inference setup.')
    print('No models, datasets, or checkpoints modified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.count <= 1500:
        parser.error('--count must be between 1 and 1500')
    run(args.count, args.device)


if __name__ == '__main__':
    main()

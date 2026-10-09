#!/usr/bin/env python3
"""SoussVoice native gold regression comparison (inference only).

Run from ~/soussvoice with existing venv activated.
Writes a new timestamped CSV; never modifies datasets or checkpoints.
"""
import argparse
import csv
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = 'vonjack/opus-mt-mul-en-big'
MODEL_PATHS = {
    'marian_original': 'experiments/language-adaptation/marian_runs/v1/best_model',
    'multilingual_v1': 'experiments/input_understanding/multilingual_lora_full_v1/best_adapter',
    'multilingual_v2': 'experiments/input_understanding/multilingual_lora_weighted_v2/best_adapter',
}

def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())

def norm_punct(text):
    text = norm(text)
    return ' '.join(re.findall(r'\w+', text, flags=re.UNICODE))

def load_pairs(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(not r.get('shi_latn') or not r.get('en') for r in rows):
        raise ValueError('Gold dataset is missing rows or shi_latn/en values')
    return rows

def check_overlap(root, gold):
    # Overlap with the user's ORIGINAL train/validation/test splits,
    # including punctuation-only differences.
    for split in ('train', 'validation', 'test'):
        path = root / 'datasets/reverse_translation/v2' / (split + '.csv')
        if not path.exists():
            print(f'Original {split} data unavailable; overlap not checked')
            continue
        with path.open(encoding='utf-8-sig', newline='') as f:
            keys = {norm_punct(r['shi_latn']) for r in csv.DictReader(f)}
        overlap = [r['shi_latn'] for r in gold if norm_punct(r['shi_latn']) in keys]
        print(f'Gold inputs overlapping original {split}: {len(overlap)}/{len(gold)}')
        for phrase in overlap[:5]:
            print('  overlap:', phrase)

def self_test():
    assert norm('  Ɣzzifɣ. ') == norm('ɣzzifɣ.')
    assert norm_punct('Ssurf-iyi...') == norm_punct('Ssurf iyi')
    assert norm_punct('Awi t.') == norm_punct('Awi t!')
    print('SELF-TEST PASSED: Unicode normalization and punctuation-equivalent checks.')
    print('No models loaded and no files modified.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--root', type=Path, default=Path.cwd())
    args = p.parse_args()
    if args.self_test:
        self_test()
        return

    import torch
    import sacrebleu
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    root = args.root.resolve()
    gold_path = root / 'datasets/native_gold_v1/native_verified.csv'
    gold = load_pairs(gold_path)
    print('Native gold regression pairs:', len(gold), flush=True)
    check_overlap(root, gold)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Device:', device, flush=True)
    results = [dict(r) for r in gold]
    for label, subpath in MODEL_PATHS.items():
        path = root / subpath
        if not path.exists():
            raise FileNotFoundError(f'Required checkpoint not found: {path}')
        print(f'\nLoading {label}...', flush=True)
        if label == 'marian_original':
            tok = AutoTokenizer.from_pretrained(str(path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        else:
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
            model = PeftModel.from_pretrained(base, str(path))
        model = model.to(device).eval()
        predictions = []
        for i, row in enumerate(gold, 1):
            inputs = tok(row['shi_latn'], return_tensors='pt', truncation=True,
                         max_length=160).to(device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, num_beams=4, max_new_tokens=80)
            pred = tok.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)
            results[i - 1][label] = pred
            print(f' {i:02d}/{len(gold)}: {pred}', flush=True)
        score = sacrebleu.corpus_chrf(predictions, [[r['en'] for r in gold]]).score
        print(f'{label} gold regression chrF: {score:.2f}', flush=True)
        del model, tok
        if device == 'mps':
            torch.mps.empty_cache()

    output_dir = root / 'experiments/input_understanding/native_gold_regression'
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = output_dir / f'comparison_{stamp}.csv'
    with output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print('\n=== REVIEW MEANING, NOT JUST chrF ===')
    for row in results:
        print('\nSHI:', row['shi_latn'])
        print('REF:', row['en'])
        for label in MODEL_PATHS:
            print(label + ':', row[label])
    print('\nSaved:', output)
    print('Training: NO. Checkpoints changed: NO. Final test split evaluated: NO.')
    print('These previously seen examples are for regression checks, NOT an independent accuracy estimate.')

if __name__ == '__main__':
    main()

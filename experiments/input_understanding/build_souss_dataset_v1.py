#!/usr/bin/env python3
"""SoussVoice Understanding Dataset V1: non-destructive dataset audit and bucketing.

Usage (from ~/soussvoice):
    python experiments/input_understanding/build_souss_dataset_v1.py --self-test
    python experiments/input_understanding/build_souss_dataset_v1.py

Input: datasets/reverse_translation/v2/{train,validation,test}.csv
Output: datasets/understanding_v1/ (new files only; refuses overwrite unless --overwrite).
No training, network, model inference, dialect certification, or edits to inputs.
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path('datasets/reverse_translation/v2')
DEST = Path('datasets/understanding_v1')
INPUT_COLUMNS = ('shi_latn', 'en')
EXTRA = ('source_norm', 'target_norm', 'alternative_count', 'dialogue_score', 'dialogue_signals', 'quality_flags')

# These are *heuristic* English-side indicators. They do not establish dialect or correctness.
FIRST_SECOND = re.compile(r"\b(?:i|i'm|i've|i'll|me|my|mine|we|we're|our|us|you|you're|you've|you'll|your|yours)\b", re.I)
INTERACTIVE = re.compile(r"\b(?:please|thanks|thank you|sorry|hello|hi|goodbye|bye|help|tell|ask|come|go|want|need|like|know|understand|feel|can|could|would|should|don't|can't|won't|where|when|why|what|who|how|which)\b", re.I)
QUESTION_START = re.compile(r"^(?:what|where|when|why|who|how|which|do|does|did|is|are|can|could|would|will|should|have|has|may)\b", re.I)


def norm(s):
    return ' '.join(unicodedata.normalize('NFC', s).casefold().split())


def surface(s):
    return ' '.join(unicodedata.normalize('NFC', s).split())


def punct_key(s):
    """Conservative near-duplicate key, for *reporting* only."""
    return re.sub(r'[\s\W_]+', '', norm(s), flags=re.UNICODE)


def dialogue_features(english):
    e = surface(english)
    words = len(e.split())
    score, signals = 0, []
    if 2 <= words <= 14:
        score += 1; signals.append('short')
    if FIRST_SECOND.search(e):
        score += 2; signals.append('personal')
    if INTERACTIVE.search(e):
        score += 2; signals.append('interactive_vocabulary')
    if '?' in e or QUESTION_START.search(e):
        score += 2; signals.append('question')
    if '!' in e:
        score += 1; signals.append('exclamation')
    return score, signals


def read_split(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rdr = csv.DictReader(f)
        if not set(INPUT_COLUMNS).issubset(rdr.fieldnames or []):
            raise ValueError(f'Missing shi_latn/en columns in {path}')
        return list(rdr), list(rdr.fieldnames)


def prepare(train, heldout):
    heldout_exact = {norm(r['shi_latn']) for r in heldout if norm(r['shi_latn'])}
    heldout_punct = {punct_key(r['shi_latn']) for r in heldout if norm(r['shi_latn'])}
    alternatives = defaultdict(set)
    for r in train:
        if norm(r['shi_latn']) and norm(r['en']):
            alternatives[norm(r['shi_latn'])].add(norm(r['en']))

    seen_pairs = set()
    result = []
    counts = Counter()
    for original in train:
        row = dict(original)
        shi, en = surface(row['shi_latn']), surface(row['en'])
        if not shi or not en:
            counts['empty_pair'] += 1
            continue
        pair = (norm(shi), norm(en))
        if pair in seen_pairs:
            counts['duplicate_pair'] += 1
            continue
        seen_pairs.add(pair)
        flags = []
        if len(shi.split()) > 30 or len(en.split()) > 40:
            flags.append('long_text_review')
        if len(shi) > 500 or len(en) > 500:
            flags.append('very_long_text_review')
        if norm(shi) in heldout_exact:
            flags.append('heldout_exact_overlap')
        elif punct_key(shi) in heldout_punct:
            flags.append('heldout_punctuation_variant_review')
        if len(alternatives[norm(shi)]) > 1:
            flags.append('multiple_valid_or_conflicting_targets_review')
        score, signals = dialogue_features(en)
        row.update({
            'source_norm': norm(shi),
            'target_norm': norm(en),
            'alternative_count': str(len(alternatives[norm(shi)])),
            'dialogue_score': str(score),
            'dialogue_signals': '|'.join(signals),
            'quality_flags': '|'.join(flags),
        })
        result.append(row)
        counts['kept'] += 1
        if score >= 4: counts['conversation_candidate'] += 1
        else: counts['general_candidate'] += 1
        for flag in flags: counts[flag] += 1

    return result, counts


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    train = [
        {'shi_latn': 'Ad ur tkcmt.', 'en': 'Do not enter.'},
        {'shi_latn': 'Ad ur tkcmt.', 'en': "Don't enter."},
        {'shi_latn': '  Ad ur tkcmt.  ', 'en': 'Do not enter.'},
        {'shi_latn': 'Mani s rad tftut azkka?', 'en': 'Where will you go tomorrow?'},
        {'shi_latn': 'Aman!', 'en': 'Water!'},
        {'shi_latn': '', 'en': 'Missing source'},
    ]
    heldout = [{'shi_latn': 'Mani s rad tftut azkka!', 'en': 'Where will you go tomorrow?'}]
    out, c = prepare(train, heldout)
    assert len(out) == 4, (len(out), c)
    assert c['duplicate_pair'] == 1 and c['empty_pair'] == 1
    assert out[0]['alternative_count'] == '2' and out[1]['alternative_count'] == '2'
    assert 'heldout_punctuation_variant_review' in out[2]['quality_flags']
    assert all(r['dialogue_score'].isdigit() for r in out)
    assert norm('  Ɣzzifɣ. ') == norm('Ɣzzifɣ.')
    assert dialogue_features('Where will you go tomorrow?')[0] >= 4
    print('SELF-TEST PASSED: alternative translations preserved; exact pairs deduplicated;')
    print('  held-out punctuation variants flagged; Unicode normalization and dialogue buckets checked.')
    print('No project datasets or models modified.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', type=Path, default=SOURCE)
    ap.add_argument('--out-dir', type=Path, default=DEST)
    ap.add_argument('--overwrite', action='store_true', help='replace previously generated output files only')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    train, columns = read_split(args.source_dir / 'train.csv')
    valid, _ = read_split(args.source_dir / 'validation.csv')
    test, _ = read_split(args.source_dir / 'test.csv')

    # Never modify the original train, validation or test CSVs.
    output = args.out_dir
    planned = [output / n for n in (
        'train_all_candidates.csv', 'train_conversation_candidates.csv',
        'train_general_candidates.csv', 'train_review_flags.csv',
        'cross_split_near_duplicates.csv', 'report.json')]
    existing = [str(p) for p in planned if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit('Output already exists; originals unchanged. Use --overwrite to regenerate:\n' + '\n'.join(existing))
    if args.source_dir.resolve() == output.resolve():
        raise SystemExit('Refusing to write into original source directory')

    rows, counts = prepare(train, valid + test)
    convo = [r for r in rows if int(r['dialogue_score']) >= 4 and not any(x in r['quality_flags'] for x in ('heldout_exact_overlap', 'heldout_punctuation_variant_review'))]
    general = [r for r in rows if int(r['dialogue_score']) < 4 and not any(x in r['quality_flags'] for x in ('heldout_exact_overlap', 'heldout_punctuation_variant_review'))]
    flagged = [r for r in rows if r['quality_flags']]
    near = [r for r in rows if 'heldout_punctuation_variant_review' in r['quality_flags'] or 'heldout_exact_overlap' in r['quality_flags']]
    fieldnames = list(dict.fromkeys(columns + list(EXTRA)))

    output.mkdir(parents=True, exist_ok=True)
    write_csv(planned[0], rows, fieldnames)
    write_csv(planned[1], convo, fieldnames)
    write_csv(planned[2], general, fieldnames)
    write_csv(planned[3], flagged, fieldnames)
    write_csv(planned[4], near, fieldnames)

    report = {
        'important': [
            'All labels are automated heuristics, NOT native-verified translations or dialect certification.',
            'Multiple English targets are preserved because they may be legitimate alternatives.',
            'Exact/near-duplicate checks do not guarantee semantic train/test independence.',
            'Existing train/validation/test datasets were read only; no model training occurred.',
            'Do not train on train_all_candidates until reviewing leakage flags; conversation/general subsets exclude flagged held-out variants.'
        ],
        'source_split_counts': {'train': len(train), 'validation': len(valid), 'test': len(test)},
        'output_counts': {'train_all_candidates': len(rows), 'train_conversation_candidates': len(convo), 'train_general_candidates': len(general), 'train_review_flags': len(flagged), 'cross_split_near_duplicates': len(near)},
        'flags_and_dedup_counts': dict(counts),
        'examples_conversation': [{'shi': r['shi_latn'], 'en': r['en'], 'score': int(r['dialogue_score'])} for r in convo[:8]],
        'examples_general': [{'shi': r['shi_latn'], 'en': r['en'], 'score': int(r['dialogue_score'])} for r in general[:8]],
    }
    planned[5].write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print('\n=== SOUSS UNDERSTANDING DATASET V1 ===')
    for k, v in report['source_split_counts'].items(): print(f'Source {k}: {v}')
    for k, v in report['output_counts'].items(): print(f'{k}: {v}')
    print(f'Exact duplicate source+English pairs removed: {counts["duplicate_pair"]}')
    print(f'Alternate English translations retained: {counts["multiple_valid_or_conflicting_targets_review"]} rows flagged for review')
    print(f'Report: {planned[5]}')
    print('Original datasets unchanged. No training performed.')


if __name__ == '__main__':
    main()

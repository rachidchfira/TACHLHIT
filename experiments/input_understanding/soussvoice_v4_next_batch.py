#!/usr/bin/env python3
"""SoussVoice V4: preserve corrections and review new conversational examples.

Offline/stdlib only. Reads existing Awal audits and native decisions; does NOT
alter them, train a model, touch checkpoints, or run reserved test inference.

User-confirmed proper-name correction for Awal:467 ('Tom') is kept in a separate
correction ledger and applied only to that record; no guesswork on other names.
New Tifinagh candidates use the Latin conversions prepared by the previous tool.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

SOURCE_LATIN = 'datasets/understanding_v4/source_audit_v1/awal_new_latin_review.csv'
SOURCE_TIF = 'datasets/understanding_v4/source_audit_v1/awal_new_tifinagh_review.csv'
ALL_LATINIZED = 'datasets/understanding_v4/tifinagh_latin_review_v1/all_tifinagh_as_latin.csv'
LATIN_REVIEW = 'datasets/understanding_v4/native_review_batch_01/native_decisions/awal_latin_review_sample.csv'
TIF_REVIEW = 'datasets/understanding_v4/tifinagh_latin_review_v1/native_decisions/latin_review_sample.csv'
BATCH = 'datasets/understanding_v4/native_review_batch_02'
CORRECTIONS = 'datasets/understanding_v4/native_corrections_v1.csv'

FIELDS = ['source_id', 'source_script', 'english', 'latin_tachelhit', 'original_tachelhit',
          'conversion_flags', 'name_watch', 'priority', 'source_url', 'license',
          'native_decision', 'corrected_latin', 'review_notes']
DECISIONS = {'a': 'accurate', 'w': 'wrong', 'u': 'uncertain', 's': 'skip'}
REVIEW_FIELDS = ('native_decision', 'corrected_latin', 'review_notes')
CORR_FIELDS = ['source_id', 'corrected_latin', 'reason', 'authority']


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required file missing: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'CSV has no header: {path}')
        return list(reader), list(reader.fieldnames)


def atomic_csv(path, items, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.souss_v4_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(items)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def norm(s):
    s = unicodedata.normalize('NFC', str(s or '')).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s).split())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_by_id(rows, label):
    ids = [r.get('source_id', '') for r in rows]
    if '' in ids or len(ids) != len(set(ids)):
        raise ValueError(f'Missing or duplicate source_id in {label}')
    return {r['source_id']: r for r in rows}


def proper_names(en):
    # Conservative flagging only: NEVER automatically rewrite names based on this.
    pieces = re.findall(r"\b[A-Z][a-z]+\b", en)
    ignore = {'I', 'The', 'A', 'An', 'He', 'She', 'We', 'You', 'They', 'It',
              'Where', 'What', 'Who', 'Why', 'When', 'How', 'Which', 'Is',
              'Are', 'Do', 'Does', 'Did', 'Can', 'Could', 'Will', 'Would',
              'Let', 'In', 'On', 'At', 'If', 'My', 'Your', 'Our', 'His', 'Her',
              'There', 'This', 'That', 'Those', 'These', 'Thank', 'Hello'}
    return ', '.join(sorted(set(x for x in pieces if x not in ignore)))


def priority(en):
    # Heuristic REVIEW priority; never a linguistic quality judgment.
    e = en.lower().strip()
    words = len(e.split())
    score = 0
    score += 6 if '?' in en else 0
    score += 4 if re.search(r"\b(i|you|we|he|she|they|my|your|our|me|us)\b", e) else 0
    score += 3 if re.search(r"\b(want|need|go|come|eat|drink|bring|tell|ask|think|know|feel|learn|help|like|love|cook|work|talk|see|saw|went|buy|can|can't|don't|didn't|not|never)\b", e) else 0
    score += 2 if 4 <= words <= 14 else 0
    score -= 8 if words <= 2 else 0
    score -= 3 if words > 19 else 0
    return score


def protected_latin_keys(root):
    protected = set()
    files = (
        'datasets/reverse_translation/v2/train.csv',
        'datasets/reverse_translation/v2/val.csv',
        'datasets/reverse_translation/v2/validation.csv',
        'datasets/reverse_translation/v2/test.csv',
        'datasets/native_gold_v1/native_verified.csv',
        'datasets/external/datashi/inputs.csv',
        'experiments/input_understanding/datashi_comparison_v1/comparisons.csv',
    )
    for rel in files:
        p = root / rel
        if p.exists():
            for row in load(p)[0]:
                for field in ('shi_latn', 'shi', 'SHI-s', 'SHI-ns'):
                    if row.get(field):
                        protected.add(norm(row[field]))
    # In these projects, at least these are required to guard held-out records.
    for rel in ('datasets/reverse_translation/v2/test.csv', 'datasets/external/datashi/inputs.csv'):
        if not (root / rel).exists():
            raise FileNotFoundError(f'Cannot check protected test data: {root / rel}')
    return protected


def ensure_correction_ledger(root, known_ids):
    p = root / CORRECTIONS
    if not p.exists():
        if 'Awal:467' not in known_ids:
            raise ValueError('Cannot apply Tom correction: source Awal:467 missing')
        # Based on user's explicit correction: leave Tifinagh original intact.
        atomic_csv(p, [{
            'source_id': 'Awal:467', 'corrected_latin': 'manimk ad iga Tom?',
            'reason': 'Native speaker: preserve proper name Tom, not ṭum',
            'authority': 'user_confirmed_spelling_correction'
        }], CORR_FIELDS)
    rows, fields = load(p)
    if not set(CORR_FIELDS).issubset(fields):
        raise ValueError(f'Corrections ledger requires fields {CORR_FIELDS}')
    corrections = unique_by_id(rows, p)
    if any(sid not in known_ids for sid in corrections):
        raise ValueError('Corrections ledger references unknown source IDs')
    return corrections, p


def build(root, n=12):
    latin, _ = load(root / SOURCE_LATIN)
    tif, _ = load(root / SOURCE_TIF)
    latinized, _ = load(root / ALL_LATINIZED)
    earlier_lat, _ = load(root / LATIN_REVIEW)
    earlier_tif, _ = load(root / TIF_REVIEW)
    lmap = unique_by_id(latin, SOURCE_LATIN)
    tmap = unique_by_id(tif, SOURCE_TIF)
    c_map = unique_by_id(latinized, ALL_LATINIZED)
    rlat = unique_by_id(earlier_lat, LATIN_REVIEW)
    rtif = unique_by_id(earlier_tif, TIF_REVIEW)
    if set(lmap) & set(tmap):
        raise ValueError('Latin and Tifinagh source IDs collide')
    if set(tmap) != set(c_map):
        raise ValueError('Converted Tifinagh source IDs differ from source audit')
    if not set(rlat).issubset(lmap) or not set(rtif).issubset(tmap):
        raise ValueError('Reviewed IDs missing from original sources')
    for sid, r in rlat.items():
        original = lmap[sid]
        if r.get('english') != original.get('english') or r.get('tachelhit') != original.get('tachelhit'):
            raise ValueError(f'Latin review source mismatch: {sid}')
    for sid, r in rtif.items():
        original = tmap[sid]
        if (r.get('english') != original.get('english') or
            r.get('tachelhit') != original.get('tachelhit') or
            r.get('transliterated_latin') != c_map[sid].get('transliterated_latin')):
            raise ValueError(f'Tifinagh review source mismatch: {sid}')
    for sid, r in c_map.items():
        original = tmap[sid]
        if r.get('english') != original.get('english') or r.get('tachelhit') != original.get('tachelhit'):
            raise ValueError(f'Tifinagh conversion mismatch: {sid}')
    corrections, ledger = ensure_correction_ledger(root, set(lmap) | set(tmap))
    protected = protected_latin_keys(root)
    reviewed = []
    collision_count = 0
    for code, earlier, master, converted in (
        ('latin', rlat, lmap, None), ('tifinagh', rtif, tmap, c_map)
    ):
        for sid, record in earlier.items():
            if record.get('native_decision') != 'accurate':
                continue
            raw = master[sid]
            proposed = raw['tachelhit'] if code == 'latin' else converted[sid]['transliterated_latin']
            correction = corrections.get(sid, {}).get('corrected_latin', '').strip()
            if correction:
                proposed = correction
            elif record.get('corrected_latin', '').strip():
                proposed = record['corrected_latin'].strip()
            elif record.get('corrected_tachelhit', '').strip():
                proposed = record['corrected_tachelhit'].strip()
            if norm(proposed) in protected:
                collision_count += 1
                # Keep visible, but never silently place in training-eligible material.
                status = 'protected_overlap_not_training_eligible'
            else:
                status = 'native_reviewed_candidate_not_training_approved'
            reviewed.append({
                'source_id': sid, 'english': raw['english'], 'shi_latn': proposed,
                'original_tachelhit': raw['tachelhit'], 'source_script': code,
                'review_decision': 'accurate', 'correction_applied': bool(correction),
                'eligibility': status, 'source_url': raw.get('source_url',''),
                'license': raw.get('license',''),
            })
    def queue(source, reviewed_ids, script):
        items = []
        for sid, record in source.items():
            if sid in reviewed_ids:
                continue
            if script == 'tifinagh':
                transformed = c_map[sid]['transliterated_latin']
                warn = c_map[sid].get('conversion_flags', '')
            else:
                transformed = record['tachelhit']
                warn = ''
            if norm(transformed) in protected:
                continue  # Skip examples now found to overlap protected sources.
            items.append({
                'source_id': sid, 'source_script': script,
                'english': record['english'], 'latin_tachelhit': transformed,
                'original_tachelhit': record['tachelhit'],
                'conversion_flags': warn,
                'name_watch': proper_names(record['english']),
                'priority': priority(record['english']),
                'source_url': record.get('source_url',''),
                'license': record.get('license',''),
                'native_decision': '', 'corrected_latin': '', 'review_notes': '',
            })
        items.sort(key=lambda r: (-int(r['priority']),
                                 hashlib.sha256(('v4-batch2|'+r['source_id']).encode()).hexdigest()))
        return items[:n], len(items)
    qlat, remaining_lat = queue(lmap, rlat, 'latin')
    qtif, remaining_tif = queue(tmap, rtif, 'tifinagh')
    reviewed.sort(key=lambda r: (r['source_script'], r['source_id']))
    report = {
        'version': 'native_review_batch_02', 'new_reviews_are_not_approved': True,
        'reviewed_accurate_v1': len(reviewed),
        'existing_review_protected_overlap': collision_count,
        'latin_remaining_eligible': remaining_lat,
        'tifinagh_remaining_eligible': remaining_tif,
        'batch_size_each': n,
        'sources_sha256': {rel: digest(root/rel) for rel in (
            SOURCE_LATIN, SOURCE_TIF, ALL_LATINIZED, LATIN_REVIEW, TIF_REVIEW)},
        'name_correction_ledger_sha256': digest(ledger),
        'notes': 'Tom manually corrected in isolated ledger. Other proper names flagged only. No automatic training approval.'
    }
    return reviewed, qlat, qtif, report


def prepare(root, out, n):
    verified, latin, tif, report = build(root, n)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding='utf-8'))
        if (old.get('sources_sha256') != report['sources_sha256'] or
            old.get('name_correction_ledger_sha256') != report['name_correction_ledger_sha256'] or
            old.get('batch_size_each') != n):
            raise ValueError('Source inputs or correction ledger changed since batch 02 preparation; refusing to overwrite.')
    for name, rows in [('latin_queue.csv', latin), ('tifinagh_latin_queue.csv', tif)]:
        path = out / name
        if path.exists():
            original, _ = load(path)
            if original != [{k: str(v) for k, v in row.items()} for row in rows]:
                raise ValueError(f'Queue changed since review began; refusing to overwrite {path}')
        else:
            atomic_csv(path, rows, FIELDS)
    approved_cols = ['source_id','english','shi_latn','original_tachelhit','source_script',
                     'review_decision','correction_applied','eligibility','source_url','license']
    verified_file = out / 'previously_reviewed_with_corrections.csv'
    if verified_file.exists():
        existing, _ = load(verified_file)
        # Preserve immutable review archive; edits require new version.
        if existing != [{k: str(v) for k,v in r.items()} for r in verified]:
            raise ValueError(f'Existing reviewed export changed: {verified_file}')
    else:
        atomic_csv(verified_file, verified, approved_cols)
    if not manifest.exists():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tom = next((r for r in verified if r['source_id']=='Awal:467'), None)
    if tom is None or tom['shi_latn'] != 'manimk ad iga Tom?':
        raise ValueError('The user-confirmed Tom correction was not carried into reviewed output')
    print('=== SOUSSVOICE V4 NEXT BATCH ===')
    print('Previously reviewed accurate:', len(verified))
    print('Tom correction:', tom['shi_latn'], '(source Awal:467)')
    print('Native reviewed pairs flagged protected:', report['existing_review_protected_overlap'])
    print(f'New prioritized review: Latin={len(latin)}, Tifinagh-as-Latin={len(tif)}')
    print('Eligible remaining before selection: Latin=', report['latin_remaining_eligible'],
          'Tifinagh-as-Latin=', report['tifinagh_remaining_eligible'])
    print('Output:', out)
    print('No training; no original changes; no test inference.')
    return latin, tif


def review(out, queue_file, rows):
    path = out/'native_decisions'/queue_file
    saved = {}
    if path.exists():
        old, _ = load(path)
        saved = unique_by_id(old, path)
        if set(saved) != {r['source_id'] for r in rows}:
            raise ValueError('Saved review IDs do not match new batch')
        for r in rows:
            prior = saved[r['source_id']]
            for field in ('source_id','english','latin_tachelhit','original_tachelhit'):
                if prior.get(field) != r.get(field):
                    raise ValueError(f'Source changed after review: {r["source_id"]}')
    working = []
    for row in rows:
        item = dict(row)
        for field in REVIEW_FIELDS:
            item[field] = saved.get(row['source_id'], {}).get(field, '')
        working.append(item)
    print(f'\n=== {queue_file} | {len(rows)} ITEMS ===')
    print('Rate meaning AND natural Latin spelling: [a] accurate, [w] wrong, [u] unsure, [s] skip, [q] quit')
    print('Proper names are warnings; only Tom/Awal:467 has a user-confirmed correction.')
    for index, item in enumerate(working, 1):
        if item['native_decision']:
            continue
        print(f'\n--- {index}/{len(working)} | {item["source_id"]} ---')
        print('ENGLISH:  ', item['english'])
        print('TACHELHIT:', item['latin_tachelhit'])
        if item['name_watch']:
            print('CHECK NAME SPELLING:', item['name_watch'])
        if item['conversion_flags']:
            print('CONVERSION WARNING:', item['conversion_flags'])
        while True:
            try:
                rating = input('Rating [a/w/u/s/q]: ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                print('\nInterrupted; previous ratings saved.')
                return False
            if rating in DECISIONS or rating == 'q':
                break
            print('Choose a/w/u/s/q')
        if rating == 'q':
            print('Stopped; resume with the same command.')
            return False
        if rating == 'a':
            # a = accepted AS SHOWN. Use w for correct meaning but spelling edits.
            pass
        item['native_decision'] = DECISIONS[rating]
        if rating == 'w':
            try:
                item['corrected_latin'] = input('Correct Latin Tachelhit (optional): ').strip()
                item['review_notes'] = input('Reason (optional): ').strip()
            except (EOFError, KeyboardInterrupt):
                print('Note entry interrupted; saving rating.')
        atomic_csv(path, working, FIELDS)
        print('Saved:', item['native_decision'])
    counts = collections.Counter(item['native_decision'] for item in working)
    print('REVIEW COMPLETE:', ', '.join(f'{k}={counts[k]}' for k in DECISIONS.values()))
    print('Review saved:', path)
    return True


def self_test():
    from tempfile import TemporaryDirectory
    assert proper_names('What is Tom doing in Paris?') == 'Paris, Tom'
    assert priority('What do you want to eat tonight?') > priority('X')
    assert norm(' Iṛmi  sul. ') == norm('iṛmi sul')
    with TemporaryDirectory() as d:
        f = Path(d)/'x.csv'
        atomic_csv(f, [{'source_id':'Awal:467','english':'What is Tom like?'}], ['source_id','english'])
        data, _ = load(f)
        assert data[0]['source_id'] == 'Awal:467'
    print('SELF-TEST PASS: safe writes, names flagged, conversation priority, Unicode normalization')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, default=Path(BATCH))
    p.add_argument('--per-source', type=int, default=12)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not 1 <= args.per_source <= 50:
        p.error('--per-source must be 1 through 50')
    root = args.root.expanduser().resolve()
    out = args.output if args.output.is_absolute() else root/args.output
    latin, tif = prepare(root, out, args.per_source)
    if not args.prepare_only:
        if not review(out, 'latin_queue.csv', latin):
            return
        review(out, 'tifinagh_latin_queue.csv', tif)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build a reproducible, provenance-preserving shi_Latn -> English dataset.
Requires: datasets (pip install datasets). No model weights are downloaded.
"""
import argparse, csv, hashlib, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = 'abdelhaqueidali/Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset'
CONFIG = 'en_shi_tifinagh'
FIELDS = ['example_id','source','source_config','source_row','source_id','shi_latn','shi_tfng','en','split']

def clean(s):
    return unicodedata.normalize('NFC', str(s or '').replace('\ufeff','').strip())

def key(s):
    return re.sub(r'\s+', ' ', clean(s).casefold()).strip()

def fingerprint(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def valid(source, target):
    return (bool(source and target) and len(source) <= 500 and len(target) <= 500
            and len(source.split()) <= 80 and len(target.split()) <= 80
            and any(c.isalpha() for c in source) and any(c.isalpha() for c in target))

def split_for(group_key):
    # Split by normalized Tachelhit source to prevent identical source appearing across splits.
    value = int(fingerprint('soussvoice-v1:'+group_key)[:12], 16) % 100
    return 'train' if value < 90 else 'validation' if value < 95 else 'test'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='datasets/reverse_translation/v1')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset(REPO, CONFIG, split='train')
    records, rejected, conflicts = [], [], []
    groups = defaultdict(list)
    stats = Counter()
    for i, raw in enumerate(ds):
        row = {'source': REPO, 'source_config': CONFIG, 'source_row': i,
               'source_id': clean(raw.get('id')),
               'shi_latn': clean(raw.get('shi_latn')),
               'shi_tfng': clean(raw.get('shi_tfng')),
               'en': clean(raw.get('en'))}
        stats['raw'] += 1
        if not valid(row['shi_latn'], row['en']):
            row['reason'] = 'empty_or_out_of_bounds'
            rejected.append(row)
            stats['rejected'] += 1
            continue
        groups[key(row['shi_latn'])].append(row)
    # Keep all source-to-target alternatives in provenance, but don't train on
    # conflicting targets without manual adjudication.
    for source_key, rows in sorted(groups.items()):
        distinct_targets = {key(r['en']) for r in rows}
        if len(distinct_targets) > 1:
            for row in rows:
                row['reason'] = 'same_tachelhit_multiple_english_targets'
                conflicts.append(row)
            stats['ambiguous_source_groups'] += 1
            continue
        canonical = rows[0].copy()
        canonical['example_id'] = fingerprint(source_key + '\t' + key(canonical['en']))[:20]
        canonical['split'] = split_for(source_key)
        records.append(canonical)
        stats['exact_or_normalized_duplicates_removed'] += len(rows) - 1
        stats[canonical['split']] += 1
    def write(name, rows, fields):
        with (out / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    write('all.csv', records, FIELDS)
    for split in ('train','validation','test'):
        write(f'{split}.csv', (r for r in records if r['split'] == split), FIELDS)
    write('rejected.csv', rejected, FIELDS + ['reason'])
    write('conflicts.csv', conflicts, FIELDS + ['reason'])
    report = dict(stats)
    report.update({'dataset':REPO,'config':CONFIG,'output':str(out),
                   'total_clean':len(records), 'conflicting_rows':len(conflicts),
                   'split_method':'SHA256 of normalized Tachelhit source; 90/5/5',
                   'limitations':['No near-duplicate semantic filtering',
                                  'No human verification of translations',
                                  'Check source dataset license and attribution before distribution',
                                  'Other datasets intentionally excluded pending audit']})
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

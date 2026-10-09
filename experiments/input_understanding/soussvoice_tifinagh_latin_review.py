#!/usr/bin/env python3
"""SoussVoice V4: review Awal Tifinagh-English candidates in Latin script.

Reads existing source audit + stable sampled queue, writes separate derived outputs.
No network, model loading, training or edits to original files.
The conversion is mechanical, not a verified translation or dialect judgment.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path

VERSION = 'tifinagh_latin_review_v1'
SAMPLE = 'awal_tifinagh_review_sample.csv'
SOURCE = 'awal_new_tifinagh_review.csv'

# IRCAM-style neo-Tifinagh letter correspondences, written in the Latin
# orthography of SoussVoice (ɛ, ɣ, c), rather than geographic-name BGN/PCGN
# notation (', gh, ch). This is transliteration, NOT translation.
MAP = {
    'ⴰ':'a', 'ⴱ':'b', 'ⴳ':'g', 'ⴷ':'d', 'ⴹ':'ḍ', 'ⴻ':'e',
    'ⴼ':'f', 'ⴽ':'k', 'ⵀ':'h', 'ⵃ':'ḥ', 'ⵄ':'ɛ', 'ⵅ':'x',
    'ⵇ':'q', 'ⵉ':'i', 'ⵊ':'j', 'ⵍ':'l', 'ⵎ':'m', 'ⵏ':'n',
    'ⵓ':'u', 'ⵔ':'r', 'ⵕ':'ṛ', 'ⵖ':'ɣ', 'ⵙ':'s', 'ⵚ':'ṣ',
    'ⵛ':'c', 'ⵜ':'t', 'ⵟ':'ṭ', 'ⵡ':'w', 'ⵢ':'y', 'ⵣ':'z',
    'ⵥ':'ẓ', 'ⵯ':'ʷ', 'ⴵ':'ǧ', 'ⵞ':'č',
}
# Nonstandard glyphs stay visible rather than being guessed.
SEPARATORS = {'⵿':' ', '⵾':' '}
DECISIONS = {'a':'accurate', 'w':'wrong', 'u':'uncertain', 's':'skip'}
REVIEW_FIELDS = ['native_decision', 'corrected_latin', 'review_notes']
DERIVED_FIELDS = ['source_id','english','tachelhit','transliterated_latin',
                  'conversion_flags','script','quality_label','source_url','license']


def read_csv(path: Path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rdr=csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError(f'CSV missing header: {path}')
        return list(rdr), list(rdr.fieldnames)


def safe_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.soussvoice_',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            wr.writeheader(); wr.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def convert(text: str):
    result=[]; warnings=[]
    for char in unicodedata.normalize('NFC',text):
        if char in MAP:
            result.append(MAP[char])
        elif char in SEPARATORS:
            result.append(SEPARATORS[char])
        elif 0x2D30 <= ord(char) <= 0x2D7F:
            result.append(f'[U+{ord(char):04X}]')
            warnings.append(f'unmapped Tifinagh U+{ord(char):04X}')
        else:
            result.append(char)
            if char.isalpha():
                warnings.append(f'non-Tifinagh letter {char!r}')
    return ''.join(result), sorted(set(warnings))


def derive(rows):
    output=[]
    for row in rows:
        for field in ('source_id','english','tachelhit'):
            if not row.get(field):
                raise ValueError(f'Required field {field} missing in {row}')
        latin, warnings=convert(row['tachelhit'])
        output.append({**row,'transliterated_latin':latin,
                       'conversion_flags':' | '.join(warnings)})
    if len({r['source_id'] for r in output}) != len(output):
        raise ValueError('Duplicate source IDs; cannot safely resume reviews')
    return output


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, out: Path):
    original_full=root/'datasets/understanding_v4/source_audit_v1'/SOURCE
    original_sample=root/'datasets/understanding_v4/native_review_batch_01'/SAMPLE
    if not original_full.is_file() or not original_sample.is_file():
        raise FileNotFoundError('Run the existing source audit and review queue preparation first.')
    full=derive(read_csv(original_full)[0]); sample=derive(read_csv(original_sample)[0])
    ids={r['source_id']:r for r in full}
    for item in sample:
        original=ids.get(item['source_id'])
        if original is None or any(item[k]!=original[k] for k in ('english','tachelhit')):
            raise ValueError('Sample no longer matches the original source audit. Stop.')
    output_full=out/'all_tifinagh_as_latin.csv'
    output_sample=out/'latin_review_sample.csv'
    report=out/'conversion_report.json'
    provenance={'version':VERSION,'training_approved':False,
                'original_audit_sha256':hash_file(original_full),
                'original_sample_sha256':hash_file(original_sample),
                'all_candidates':len(full),'review_batch':len(sample),
                'conversion_warnings_all':sum(bool(x['conversion_flags']) for x in full),
                'conversion_warnings_sample':sum(bool(x['conversion_flags']) for x in sample),
                'note':'Latin letter conversion is mechanical, NOT native-verified. The Tifinagh original is preserved.'}
    # Fail on a changed source, instead of silently overwriting a review in progress.
    if report.exists():
        old=json.loads(report.read_text(encoding='utf-8'))
        if old.get('original_audit_sha256') != provenance['original_audit_sha256'] or old.get('original_sample_sha256') != provenance['original_sample_sha256']:
            raise ValueError('Source inputs changed since review began; refusing to overwrite.')
    out.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(DERIVED_FIELDS + [k for r in full+sample for k in r]))
    # Existing generated files are checked for changes to prevent accidental resets.
    for path,data in ((output_full,full),(output_sample,sample)):
        if path.exists():
            existing,_=read_csv(path)
            if len(existing)!=len(data) or any(any(a.get(k,'')!=b.get(k,'') for k in fields) for a,b in zip(existing,data)):
                raise ValueError(f'Existing derived data differs, refusing overwrite: {path}')
        else:
            safe_csv(path,data,fields)
    if not report.exists():
        report.write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('=== SOUSSVOICE TIFINAGH -> LATIN ===')
    print(f'Converted full source: {len(full)}; review sample: {len(sample)}')
    print(f'Flagged conversions: {provenance["conversion_warnings_all"]} full, {provenance["conversion_warnings_sample"]} sample')
    print('Preview (English -> Latin):')
    for r in sample[:4]:
        print('  ',r['english'],'->',r['transliterated_latin'])
        if r['conversion_flags']: print('     CONVERSION WARNING:',r['conversion_flags'])
    print('Output:',out)
    print('Original Tifinagh preserved. No training and no native approval assumed.')
    return sample


def review(out: Path, sample: list[dict], show_original=False):
    destination=out/'native_decisions'/'latin_review_sample.csv'
    if destination.exists():
        old, _=read_csv(destination)
        saved={r['source_id']:r for r in old}
        if len(saved)!=len(old) or set(saved)!={r['source_id'] for r in sample}:
            raise ValueError('Saved review IDs differ from original sample; stop.')
        for r in sample:
            previous=saved[r['source_id']]
            for k in ('english','tachelhit','transliterated_latin'):
                if r.get(k)!=previous.get(k):
                    raise ValueError('Source or conversion changed mid-review; stop.')
    else:
        saved={}
    working=[]
    for row in sample:
        old=saved.get(row['source_id'],{})
        item=dict(row)
        for field in REVIEW_FIELDS:
            item[field]=old.get(field,'')
        working.append(item)
    fieldnames=list(dict.fromkeys([k for row in working for k in row]+REVIEW_FIELDS))
    ratings=collections.Counter(r['native_decision'] for r in working)
    print(f'\n=== LATIN-SCRIPT NATIVE REVIEW: {len(working)} SENTENCES ===')
    print(f'Previously reviewed: {len(working)-ratings[""]}')
    print('[a] Correct meaning AND natural Latin Tachelhit')
    print('[w] Wrong meaning, spelling or dialect (give correction if possible)')
    print('[u] Unsure   [s] Skip   [q] Quit safely')
    for n,item in enumerate(working,1):
        if item['native_decision']: continue
        print(f'\n--- {n}/{len(working)} | {item["source_id"]} ---')
        print('ENGLISH:  ',item['english'])
        print('TACHELHIT:',item['transliterated_latin'])
        if item['conversion_flags']:
            print('AUTOMATIC CONVERSION WARNING:',item['conversion_flags'])
        if show_original:
            print('ORIGINAL TIFINAGH:',item['tachelhit'])
        while True:
            try:
                key=input('Rating [a/w/u/s/q]: ').strip().lower()
            except (EOFError,KeyboardInterrupt):
                print('\nInterrupted. Previous decisions saved.')
                return
            if key in DECISIONS or key=='q':break
            print('Please type a, w, u, s, or q.')
        if key=='q':
            print('Stopped. You can resume with the same command.')
            return
        item['native_decision']=DECISIONS[key]
        if key=='w':
            try:
                item['corrected_latin']=input('Better Latin Tachelhit (optional): ').strip()
                item['review_notes']=input('Reason (optional): ').strip()
            except (EOFError,KeyboardInterrupt):
                print('\nSaving review; note entry interrupted.')
        safe_csv(destination,working,fieldnames)
        print('Saved:',item['native_decision'])
    counts=collections.Counter(r['native_decision'] for r in working)
    print('\n=== TIFINAGH -> LATIN REVIEW COMPLETE ===')
    print(', '.join(f'{v}={counts[v]}' for v in ('accurate','wrong','uncertain','skip')))
    print('Native decisions:',destination)
    print('Original data and held-out sets unchanged. No training.')


def self_test():
    fixtures={
        'ⴰⵣⵓⵍ':'azul',
        'ⴰⴳⴰⴷⵉⵔ':'agadir',
        'ⵙⵉⵎⵔⴰⵡ':'simraw',
        'ⴽⴽⵓⵎⵔⴰⵡ ⴷ ⵜⵥⴰ':'kkumraw d tẓa',
        'ⴰⵥⴳⴳⵯⴰⵖ':'aẓggʷaɣ',
        'ⵉⴽⵙ':'iks',
        'ⵓⵔ ⵔⵉⵖ':'ur riɣ',
    }
    for a,b in fixtures.items():
        got,warn=convert(a)
        assert got==b and not warn,(a,got,warn)
    assert convert('⵿ⴰ')[0]==' a'
    assert convert('⵿ⵠ')[1], 'Unknown Tifinagh must be flagged'
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        file=root/'x.csv'
        rows=[{'source_id':'1','english':'Hello','tachelhit':'ⴰⵣⵓⵍ'}]
        safe_csv(file,rows,list(rows[0]))
        assert read_csv(file)[0]==rows
        assert derive(rows)[0]['transliterated_latin']=='azul'
    print('SELF-TEST PASS: orthography, gemination, unknown flags, CSV persistence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--show-original',action='store_true')
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,default=Path('datasets/understanding_v4/tifinagh_latin_review_v1'))
    args=parser.parse_args()
    if args.self_test:return self_test()
    root=args.root.expanduser().resolve()
    out=args.output if args.output.is_absolute() else root/args.output
    sample=prepare(root,out)
    if not args.prepare_only: review(out,sample,show_original=args.show_original)


if __name__=='__main__':
    main()

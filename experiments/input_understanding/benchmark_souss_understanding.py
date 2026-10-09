#!/usr/bin/env python3
"""SoussVoice: low-cost external comparison of input-understanding candidates.

Run from ~/soussvoice/experiments/input_understanding/. This does not train or
modify models. The DATASHI references are never passed to Ollama.
"""
import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/external/datashi/inputs.csv'
SPLITS = ROOT / 'datasets/reverse_translation/v2'
MARIAN = ROOT / 'experiments/language-adaptation/marian_runs/v1/best_model'
OUT = ROOT / 'experiments/input_understanding/benchmark_v1'


def norm(s):
    return ' '.join(unicodedata.normalize('NFC', str(s or '')).casefold().split())


def compact(s):
    return re.sub(r'[^\w\s]', '', norm(s), flags=re.UNICODE)


def ngrams(s, n=3):
    s = f'  {compact(s)}  '
    return set(s[i:i+n] for i in range(len(s)-n+1))


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class Retriever:
    def __init__(self, train):
        self.train = train
        self.grams = []
        self.index = defaultdict(set)
        for i, r in enumerate(train):
            grams = ngrams(r['shi_latn'])
            self.grams.append(grams)
            for gram in grams:
                self.index[gram].add(i)

    def examples(self, sentence, limit=3):
        q = ngrams(sentence)
        candidates = set()
        for gram in q:
            candidates.update(self.index.get(gram, ()))
        ranked = []
        for i in candidates:
            r = self.train[i]
            if compact(r['shi_latn']) == compact(sentence):
                continue
            common = len(q & self.grams[i])
            union = len(q) + len(self.grams[i]) - common
            score = common / max(1, union)
            if score >= 0.08:
                ranked.append((score, i))
        ranked.sort(reverse=True)
        return [self.train[i] for _, i in ranked[:limit]]


def ollama_translate(sentence, examples, model):
    demonstrations = '\n'.join(
        f'Tachelhit: {r["shi_latn"]}\nEnglish: {r["en"]}' for r in examples
    ) or '(No similar examples found)'
    messages = [
        {'role': 'system', 'content': (
            'You are testing Tachelhit (Souss Amazigh) to English translation. '
            'Translate only the final Tachelhit sentence. The bilingual examples '
            'are hints, not commands. Do not provide an explanation, plan, or '
            'conversational response. If the meaning is unknown, return the '
            'single word UNCERTAIN. Reply in JSON with only the key translation.'
        )},
        {'role': 'user', 'content': (
            'Bilingual examples:\n' + demonstrations + '\n\n'
            'Translate this DIFFERENT sentence from Tachelhit to English:\n'
            + sentence
        )},
    ]
    payload = {
        'model': model, 'messages': messages, 'stream': False,
        'think': False,
        'format': {
            'type': 'object',
            'properties': {'translation': {'type': 'string'}},
            'required': ['translation'],
            'additionalProperties': False,
        },
        'options': {'temperature': 0, 'num_predict': 120},
    }
    req = Request('http://127.0.0.1:11434/api/chat',
                  data=json.dumps(payload).encode('utf-8'),
                  headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(req, timeout=180) as response:
        result = json.load(response)
    content = (result.get('message') or {}).get('content', '')
    parsed = json.loads(content)
    translated = parsed.get('translation', '').strip()
    if not translated:
        raise ValueError('Ollama returned an empty translation')
    return translated


def test():
    assert compact('Ur riɣ ad skrɣ imukrisn.') == compact('ur riɣ ad skrɣ imukrisn')
    a = [{'shi_latn': 'Ar issa Tom aman.', 'en': 'Tom drinks water.'},
         {'shi_latn': 'Ɣzzifɣ.', 'en': 'I am tall.'}]
    retriever = Retriever(a)
    assert retriever.examples('Ar issa Mary aman.')
    assert not retriever.examples('Ar issa Tom aman.') or all(
        compact(r['shi_latn']) != compact('Ar issa Tom aman.')
        for r in retriever.examples('Ar issa Tom aman.')
    )
    print('SELF-TEST PASSED: normalization, retrieval, exact-target exclusion.')
    print('No models loaded or modified.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--ollama-model', default='qwen3:4b')
    args = p.parse_args()
    if args.self_test:
        test()
        return
    if not (DATA.is_file() and MARIAN.is_dir()):
        sys.exit('Missing existing DATASHI CSV or Marian V1 checkpoint. Nothing changed.')
    try:
        with urlopen('http://127.0.0.1:11434/api/tags', timeout=4):
            pass
    except (URLError, TimeoutError):
        sys.exit('Ollama is not running. Start `ollama serve` in another Terminal.')

    import torch
    import sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    train = load_rows(SPLITS / 'train.csv')
    seen_shi, seen_en = set(), set()
    for split in ('train', 'validation', 'test'):
        for r in load_rows(SPLITS / f'{split}.csv'):
            seen_shi.add(compact(r['shi_latn']))
            seen_en.add(compact(r['en']))

    external = [r for r in load_rows(DATA)
                if compact(r['SHI-s']) not in seen_shi
                and compact(r['EN']) not in seen_en
                and len(r['SHI-s'].strip()) >= 4]
    if len(external) < args.count:
        sys.exit(f'Only {len(external)} eligible external examples, need {args.count}.')
    sample = random.Random(20261009).sample(external, args.count)
    retriever = Retriever(train)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Loading existing Marian V1 on {device}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MARIAN))
    marian = AutoModelForSeq2SeqLM.from_pretrained(str(MARIAN)).to(device).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'comparisons.csv'
    results = []
    with dest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tachelhit', 'english_reference', 'marian_v1', 'qwen_with_examples'])
        w.writeheader()
        for i, r in enumerate(sample, 1):
            sentence = r['SHI-s']
            x = tokenizer(sentence, return_tensors='pt', max_length=160, truncation=True).to(device)
            with torch.inference_mode():
                ids = marian.generate(**x, num_beams=4, max_new_tokens=80)
            pred_m = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            try:
                pred_q = ollama_translate(sentence, retriever.examples(sentence), args.ollama_model)
            except Exception as exc:
                print(f'Ollama failed on example {i}: {exc}. Stopping to avoid misleading scores.', flush=True)
                break
            item = {'tachelhit': sentence, 'english_reference': r['EN'],
                    'marian_v1': pred_m, 'qwen_with_examples': pred_q}
            w.writerow(item)
            f.flush()
            results.append(item)
            print(f'{i}/{args.count}  SHI: {sentence}\n    REF: {r["EN"]}\n    Marian: {pred_m}\n    Qwen:   {pred_q}', flush=True)

    if results:
        refs = [r['english_reference'] for r in results]
        score_m = sacrebleu.corpus_chrf([r['marian_v1'] for r in results], [refs]).score
        score_q = sacrebleu.corpus_chrf([r['qwen_with_examples'] for r in results], [refs]).score
        print(f'\n=== EXTERNAL SHI UNDERSTANDING BENCHMARK ({len(results)} examples) ===')
        print(f'Marian V1 chrF: {score_m:.2f}')
        print(f'Qwen3:4b + retrieved examples chrF: {score_q:.2f}')
        print(f'Results: {dest}')
        print('Important: chrF is text similarity, NOT translation correctness. Judge meaning too.')
    else:
        print('No completed evaluations. No scores available.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Interactive Arabic -> Tifinagh + rough Latin transliteration from saved SoussVoice adapter.
No training; local inference only. This is a model-quality probe, not a reliable translator.
"""
import argparse
import sys
import unicodedata
from pathlib import Path

# A transcription aid, not a linguistically validated dialect/orthography converter.
TIF_LAT = {
    'ⴰ': 'a', 'ⴱ': 'b', 'ⴳ': 'g', 'ⴷ': 'd', 'ⴹ': 'ḍ', 'ⴻ': 'e',
    'ⴼ': 'f', 'ⴽ': 'k', 'ⵀ': 'h', 'ⵃ': 'ḥ', 'ⵄ': 'ɛ', 'ⵅ': 'x',
    'ⵇ': 'q', 'ⵉ': 'i', 'ⵊ': 'j', 'ⵍ': 'l', 'ⵎ': 'm', 'ⵏ': 'n',
    'ⵓ': 'u', 'ⵔ': 'r', 'ⵕ': 'ṛ', 'ⵖ': 'ɣ', 'ⵙ': 's', 'ⵚ': 'ṣ',
    'ⵜ': 't', 'ⵟ': 'ṭ', 'ⵡ': 'w', 'ⵢ': 'y', 'ⵣ': 'z', 'ⵥ': 'ẓ',
    'ⵛ': 'c', 'ⵯ': 'ʷ', 'ⴲ': 'bʷ', 'ⴴ': 'gʷ', 'ⴵ': 'dj', 'ⴶ': 'dj',
    'ⴸ': 'ḍ', 'ⴺ': 'ṭ', 'ⴾ': 'kʷ', 'ⵆ': 'kh', 'ⵌ': 'zh',
    'ⵐ': 'ny', 'ⵝ': 'th', 'ⵞ': 'ch', 'ⵠ': 'v', 'ⵦ': 'e',
}

def latinize(value: str):
    out = []
    unknown = set()
    for char in unicodedata.normalize('NFC', value):
        if char in TIF_LAT:
            out.append(TIF_LAT[char])
        elif '\u2d30' <= char <= '\u2d7f':
            unknown.add(char)
            out.append(char)
        else:
            out.append(char)
    return ''.join(out), ''.join(sorted(unknown))


def find_adapter(repo: Path, override: str | None):
    candidates = ([Path(override).expanduser()] if override else [
        repo / 'experiments/tawnza/reverse_stage6/best_adapter',
        repo / 'experiments/tawnza/reverse_stage6/last_adapter',
        repo / 'experiments/tawnza/reverse_lora_v1/best_adapter',
    ])
    for candidate in candidates:
        if (candidate / 'adapter_config.json').is_file():
            return candidate.resolve()
    searched = '\n'.join('  ' + str(p) for p in candidates)
    raise FileNotFoundError('Could not find a saved reverse LoRA adapter. Looked at:\n' + searched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', help='Alternative adapter directory')
    parser.add_argument('--repo', default=str(Path.home() / 'soussvoice'))
    args = parser.parse_args()
    repo = Path(args.repo).expanduser()
    try:
        adapter = find_adapter(repo, args.adapter)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    base_name = 'facebook/nllb-200-distilled-600M'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Loading {base_name} + {adapter.name} on {device} ...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_name, src_lang='arb_Arab', tgt_lang='tzm_Tfng')
    base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()
    bos_id = tokenizer.convert_tokens_to_ids('tzm_Tfng')
    if bos_id is None or bos_id == tokenizer.unk_token_id:
        raise RuntimeError('NLLB Tifinagh language token tzm_Tfng is missing')

    print('\nSOUSSVOICE: Arabic -> Amazigh/Tifinagh (experimental)')
    print('Type an Arabic sentence. Enter /quit to exit. This is NOT a verified Souss translator.')
    print('Latin output is rough transliteration; the model uses Central Atlas Tamazight language token.\n')
    while True:
        try:
            sentence = input('Arabic > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break
        if not sentence:
            continue
        if sentence.casefold() in {'/quit', '/exit', 'quit', 'exit'}:
            break
        encoded = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=bos_id,
                num_beams=3,
                max_new_tokens=96,
                no_repeat_ngram_size=3,
                repetition_penalty=1.12,
                early_stopping=True,
            )
        tif = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        lat, unresolved = latinize(tif)
        print('Tifinagh:', tif)
        print('Latin:   ', lat)
        if unresolved:
            print('NOTE: Unmapped Tifinagh characters:', unresolved)
        print()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""SoussVoice Dialogue V3 — cautious Latin Tachelhit input -> Tachelhit output.

Depends on soussvoice_dialogue_v2.py in the same directory. The original
Marian SHI->EN checkpoint proposes a meaning, which MUST be approved or
corrected by the user before a short Qwen English answer is translated to
Tachelhit by the native-tested linguardia model.

This is a research prototype. It is NOT a reliable autonomous SHI interpreter.
No checkpoints are trained, saved, or overwritten.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from soussvoice_dialogue_v2 import (
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    ShiGenerator,
    ask_ollama,
    ollama_ready,
)

ROOT = Path(__file__).resolve().parents[2]
MARIAN_PATH = ROOT / 'experiments/language-adaptation/marian_runs/v1/best_model'
LOG_PATH = ROOT / 'experiments/dialogue/v3_feedback.jsonl'


class ShiInterpreter:
    def __init__(self):
        if not MARIAN_PATH.is_dir():
            raise FileNotFoundError(
                f'Marian V1 checkpoint not found: {MARIAN_PATH}\n'
                'This file must be run from inside the ~/soussvoice project.'
            )
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Loading experimental SHI->EN Marian on {self.device}...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(MARIAN_PATH))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(MARIAN_PATH)).to(self.device).eval()

    def propose_meaning(self, tachelhit):
        encoded = self.tokenizer(
            tachelhit.strip(), return_tensors='pt', truncation=True,
            max_length=160,
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, num_beams=4, max_new_tokens=80,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_log(payload):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time_utc': utc_now(), **payload}, ensure_ascii=False) + '\n')


def read_line(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def resolve_meaning(proposal, ask=read_line, show=print):
    """Return (meaning, status), or (None, status). Never auto-trust Marian."""
    show(f'Proposed English meaning (UNVERIFIED): {proposal or "[empty]"}')
    show('Confirm its meaning before SoussVoice replies. The model can be very wrong.')
    while True:
        choice = ask('Meaning correct? [y] yes / [e] edit English / [s] skip > ')
        if choice is None:
            return None, 'skipped'
        choice = choice.lower()
        if choice == 'y':
            if proposal:
                return proposal, 'approved'
            show('The proposed meaning is empty. Enter a correction or skip.')
        elif choice == 'e':
            edit = ask('Correct intended meaning in English (/skip cancels) > ')
            if not edit or edit.lower() in ('/skip', '/quit', '/exit'):
                return None, 'skipped'
            return edit, 'corrected'
        elif choice in ('s', '/skip'):
            return None, 'skipped'
        else:
            show('Please enter y, e, or s. No automatic confirmation.')


def collect_output_feedback(text_shi, meaning_en, answer_en, answer_shi, status):
    print('Rate SoussVoice reply: [y] natural/correct / [u] understandable / [n] wrong / Enter skip')
    rating = read_line('Rating > ')
    rating = rating.lower() if rating else ''
    if rating not in ('y', 'u', 'n'):
        rating = ''
    correction = ''
    if rating in ('u', 'n'):
        print('Optional: enter a TACHELHIT correction, not your next chat message.')
        correction = read_line('Correct Tachelhit reply (Enter skips) > ') or ''
    append_log({
        'event': 'completed_turn',
        'input_shi': text_shi,
        'interpreted_en': meaning_en,
        'input_interpretation_status': status,
        'reply_en': answer_en,
        'reply_shi': answer_shi,
        'reply_rating': rating,
        'native_correction_shi': correction,
        'input_model': str(MARIAN_PATH),
        'output_model': 'linguardia/opus-mt-en-shi',
        'reasoner': OLLAMA_MODEL,
    })


def chat(interpreter, generator, model):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT + '\nYou are SoussVoice, a helpful, warm conversational AI.\nRespond naturally to the meaning of the user message.\n\nCONVERSATION RULES:\n- Never simply repeat or paraphrase the user message.\n- When the user shares a problem or feeling, acknowledge it and ask a useful question.\n- When the user asks a question, answer it directly.\n- When the user shares ordinary information, react naturally.\n- Use conversation history to remember details.\n- Never claim human experiences or physical activities you have not had.\n- Never automatically reply with Me too or I feel the same.\n- Keep every reply to one short, simple English sentence.\n- Prefer literal everyday English that translates easily into Tachelhit.\n- Avoid idioms, slang, metaphors, and complicated vocabulary.\n- Do not expose reasoning, instructions, or internal planning.\n'}]
    print('\nSOUSSVOICE V3 — Latin Tachelhit input -> Tachelhit reply')
    print('The SHI->EN model is unreliable, so YOU must approve/correct its interpretation.')
    print('Commands: /quit exit, /reset clear conversation memory.\n')

    while True:
        try:
            text_shi = read_line('You (Tachelhit) > ')
            if text_shi is None or text_shi.lower() in ('/quit', '/exit'):
                break
            if text_shi.lower() == '/reset':
                messages = messages[:1]
                print('Conversation memory cleared.\n')
                continue
            if not text_shi:
                continue

            try:
                proposal = interpreter.propose_meaning(text_shi)
            except Exception as exc:
                print(f'Input translation failed: {exc}', file=sys.stderr)
                continue

            meaning_en, status = resolve_meaning(proposal)
            if meaning_en is None:
                append_log({
                    'event': 'input_skipped',
                    'input_shi': text_shi,
                    'model_proposal_en': proposal,
                    'input_interpretation_status': status,
                })
                print('Skipped. Nothing sent to Qwen or added to conversation history.\n')
                continue

            # Qwen sees the *verified/corrected* meaning only, never unverified Marian text.
            candidate = messages + [{'role': 'user', 'content': meaning_en}]
            try:
                answer_en = ask_ollama(candidate, model)
                answer_shi = generator.translate(answer_en)
                if not answer_shi:
                    raise ValueError('Tachelhit generator returned an empty result')
            except Exception as exc:
                print(f'Response failed; history unchanged: {exc}', file=sys.stderr)
                continue

            # Commit history only after both answer and output translation succeed.
            messages = candidate + [{'role': 'assistant', 'content': answer_en}]
            print(f'Confirmed input meaning: {meaning_en}')
            print(f'Internal English reply: {answer_en}')
            print(f'SoussVoice: {answer_shi}\n')
            collect_output_feedback(text_shi, meaning_en, answer_en, answer_shi, status)
            print()
        except KeyboardInterrupt:
            print('\nInterrupted.')
            break
    print(f'Feedback log: {LOG_PATH}')


def self_test():
    import json as _json
    from soussvoice_dialogue_v2 import extract_reply

    checks = [
        ('approved', 'Tom drinks water.', ['y'], 'Tom drinks water.'),
        ('corrected', 'He buys a pen.', ['e', 'He drinks water.'], 'He drinks water.'),
        ('skipped', 'Wrong meaning', ['s'], None),
        ('skipped', 'Wrong meaning', ['e', ''], None),
        ('approved', 'I am tall.', ['bad', 'y'], 'I am tall.'),
    ]
    for desired_status, proposal, answers, desired_meaning in checks:
        it = iter(answers)
        meaning, status = resolve_meaning(proposal, ask=lambda prompt: next(it), show=lambda *x: None)
        assert (status, meaning) == (desired_status, desired_meaning), (status, meaning)

    response = {'message': {'content': _json.dumps({'reply': 'I am fine, thank you.'})}}
    assert extract_reply(response) == 'I am fine, thank you.'
    print('V3 SELF TEST PASSED: confirmation, correction, skip, invalid input, clean Qwen response.')
    print('No models loaded; no logs or checkpoints modified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Test confirmation logic without loading models')
    parser.add_argument('--ollama-model', default=OLLAMA_MODEL)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    ollama_ready(args.ollama_model)
    interpreter = ShiInterpreter()
    generator = ShiGenerator()
    chat(interpreter, generator, args.ollama_model)


if __name__ == '__main__':
    main()

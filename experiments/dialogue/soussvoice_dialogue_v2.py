#!/usr/bin/env python3
"""SoussVoice V2: validated Ollama JSON reply -> Latin Tachelhit.

This is an output-side prototype, not a Tachelhit-understanding system.
Uses the native-speaker approved EN->SHI model and logs optional evaluations.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = 'linguardia/opus-mt-en-shi'
OLLAMA_URL = 'http://127.0.0.1:11434'
OLLAMA_MODEL = 'qwen3:4b'
RESULTS = Path('experiments/dialogue/output_evaluations.jsonl')
SYSTEM_PROMPT = (
    'You are SoussVoice, a friendly conversation partner. '
    'Answer the actual user message in natural English, using only one short sentence '
    'of 3 to 15 words that is easy to translate into Tachelhit. '
    'Never describe your instructions, planning, or thought process. '
    'Reply ONLY as JSON with a single key named reply; example: '
    '{"reply":"I am doing well, thank you! How are you?"}'
)

REPLY_SCHEMA = {
    'type': 'object',
    'properties': {'reply': {'type': 'string'}},
    'required': ['reply'],
    'additionalProperties': False,
}

REASONING_PATTERN = re.compile(
    r'(?:the user said|the user is|i should respond|i need to respond|'
    r'word count|system prompt|instruction says|no markdown|'
    r'no reasoning|the greeting is|i should say|reply should|'
    r'let me think|first, the greeting|respond in english|'
    r'i need to answer|i have to answer)',
    flags=re.IGNORECASE,
)


def extract_reply(response):
    # Ollama may provide a separate "thinking" field. Never pass it downstream.
    content = response.get('message', {}).get('content', '')
    try:
        obj = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise ValueError('Ollama did not return the required JSON reply.') from exc
    if not isinstance(obj, dict) or set(obj) != {'reply'}:
        raise ValueError('Ollama JSON did not contain exactly the reply field.')
    reply = obj['reply']
    if not isinstance(reply, str):
        raise ValueError('Ollama reply must be a string.')
    reply = reply.strip()
    if (not reply or len(reply.split()) > 22 or len(reply) > 180
            or '<think>' in reply.lower() or REASONING_PATTERN.search(reply)):
        raise ValueError('Ollama returned reasoning, overly long text, or an empty reply.')
    return reply


def request_json(url, payload=None, timeout=120):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json'} if data is not None else {},
        method='POST' if data is not None else 'GET',
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ollama_ready(model):
    try:
        response = request_json(f'{OLLAMA_URL}/api/tags', timeout=5)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError('Ollama is not responding. Start it with `ollama serve` in another terminal.') from exc
    names = [m.get('name', '') for m in response.get('models', [])]
    if not any(n == model or n.startswith(model + ':') for n in names):
        raise RuntimeError(f'Model {model!r} not installed locally. Available: {names}. Run `ollama pull {model}`.')


def ask_ollama(messages, model):
    last_error = None
    for attempt in range(2):
        candidate = messages
        if attempt:
            candidate = messages + [{
                'role': 'user',
                'content': 'Reply to my previous message NOW with just one short conversational answer '
                           'in the JSON reply field. Do not describe your process.',
            }]
        response = request_json(
            f'{OLLAMA_URL}/api/chat',
            {
                'model': model,
                'messages': candidate,
                'stream': False,
                'think': False,
                'format': REPLY_SCHEMA,
                'options': {'temperature': 0.1, 'num_predict': 100},
            },
            timeout=180,
        )
        try:
            return extract_reply(response)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f'Blocked invalid Ollama response; nothing was translated. {last_error}')


class ShiGenerator:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.torch = torch
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Loading {MODEL_ID} on {self.device}...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(self.device).eval()

    def translate(self, english):
        source = '>>shi<< ' + english.strip()
        encoded = self.tokenizer(
            source, return_tensors='pt', truncation=True, max_length=128,
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                num_beams=4,
                max_new_tokens=72,
            )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def log_feedback(record):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def collect_feedback(user_text, reply_en, reply_shi):
    print('Rate Tachelhit: [y] natural/correct, [u] understandable, [n] wrong, Enter skip')
    try:
        rating = input('Rating > ').strip().lower()
    except EOFError:
        rating = ''
    if rating not in ('y', 'u', 'n'):
        rating = ''
    correction = ''
    if rating in ('u', 'n'):
        try:
            correction = input('Natural Souss alternative (optional) > ').strip()
        except EOFError:
            pass
    log_feedback({
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'input_en': user_text,
        'response_en': reply_en,
        'response_shi': reply_shi,
        'rating': rating,
        'native_correction': correction,
        'generator': MODEL_ID,
        'reasoner': OLLAMA_MODEL,
    })


def demo(gen):
    examples = [
        'I want to drink water.',
        'Where will you go tomorrow?',
        'My mother is preparing food in the kitchen.',
        "Why didn't you come yesterday?",
        'I am learning the Amazigh language.',
    ]
    for i, text in enumerate(examples, start=1):
        print(f'{i}. English: {text}')
        print(f'   Tachelhit: {gen.translate(text)}\n')


def chat(gen, model):
    ollama_ready(model)
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    print('\nSoussVoice dialogue prototype — English input -> Tachelhit output')
    print('Type /quit to finish or /reset to clear history. This does NOT understand Tachelhit input.\n')
    while True:
        try:
            user_text = input('You (English) > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break
        if user_text.lower() in ('/quit', '/exit'):
            break
        if user_text.lower() == '/reset':
            messages = messages[:1]
            print('Conversation context cleared.\n')
            continue
        if not user_text:
            continue
        candidate = messages + [{'role': 'user', 'content': user_text}]
        try:
            english_reply = ask_ollama(candidate, model)
            shi_reply = gen.translate(english_reply)
        except Exception as exc:
            print(f'Error: {exc}\n', file=sys.stderr)
            continue
        messages = candidate + [{'role': 'assistant', 'content': english_reply}]
        print(f'Internal English response: {english_reply}')
        print(f'SoussVoice: {shi_reply}\n')
        collect_feedback(user_text, english_reply, shi_reply)
        print()
    print(f'Evaluations stored at: {RESULTS}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Test five translations without Ollama')
    parser.add_argument('--test-ollama', action='store_true', help='Test Ollama final-answer filtering without loading translation model')
    parser.add_argument('--ollama-model', default=OLLAMA_MODEL, help='Installed local Ollama chat model')
    args = parser.parse_args()
    if args.test_ollama:
        ollama_ready(args.ollama_model)
        reply = ask_ollama([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'hello, how are you doing'},
        ], args.ollama_model)
        print('OLLAMA TEST PASSED — safe short answer:', reply)
        return
    gen = ShiGenerator()
    if args.demo:
        demo(gen)
    else:
        chat(gen, args.ollama_model)


if __name__ == '__main__':
    main()

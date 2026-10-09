"""SoussVoice: bounded 8-example memorization test of the existing NLLB V2 adapter.

Does not modify any existing adapters or corpora. Outputs results to an isolated folder.
"""
import csv
import json
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = 'facebook/nllb-200-distilled-600M'
ADAPTER = Path('experiments/tawnza/lora_pilot_v2/adapter')
DATA = Path('datasets/external/tawnza/clean_v2/train.parquet')
OUT = Path('experiments/tawnza/memorization_diagnostic')
OUT.mkdir(parents=True, exist_ok=True)
assert (ADAPTER / 'adapter_config.json').is_file(), f'Missing V2 adapter at {ADAPTER}'
assert DATA.is_file(), f'Missing training dataset at {DATA}'

torch.manual_seed(42)
torch.set_num_threads(4)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('Device:', device, flush=True)

rows = pq.read_table(DATA, columns=['source', 'target']).to_pylist()
random.Random(2026).shuffle(rows)
# Choose moderate-sized bilingual pairs from the same shuffled order V2 used.
examples = [r for r in rows[:1600] if 15 <= len(r['target']) <= 90 and 10 <= len(r['source']) <= 100][:8]
assert len(examples) == 8

tok = AutoTokenizer.from_pretrained(BASE, src_lang='tzm_Tfng', tgt_lang='arb_Arab')
arabic_id = tok.convert_tokens_to_ids('arb_Arab')
assert arabic_id != tok.unk_token_id, 'Arabic language ID not recognized'
base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=True).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable parameters:', trainable, flush=True)
assert trainable > 0, 'LoRA weights are frozen'

src = tok([r['target'] for r in examples], padding=True, truncation=True,
          max_length=96, return_tensors='pt').to(device)
targets = tok(text_target=[r['source'] for r in examples], padding=True,
              truncation=True, max_length=96, return_tensors='pt')['input_ids'].to(device)
targets[targets == tok.pad_token_id] = -100
refs = [r['source'] for r in examples]


def evaluate():
    model.eval()
    model.config.use_cache = True
    preds = []
    for start in (0, 4):
        batch = {k: v[start:start+4] for k, v in src.items()}
        with torch.inference_mode():
            tokens = model.generate(**batch, forced_bos_token_id=arabic_id,
                                    num_beams=3, max_new_tokens=80)
        preds.extend(tok.batch_decode(tokens, skip_special_tokens=True))
    exact = sum(p.strip() == r.strip() for p, r in zip(preds, refs))
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return preds, exact, chrf


def report(step):
    preds, exact, chrf = evaluate()
    print(f'CHECK step={step} exact={exact}/8 chrF={chrf:.2f}', flush=True)
    history.append({'step': step, 'exact': exact, 'chrF': round(chrf, 2)})
    if step in (0, 40, 80, 160):
        for i in range(2):
            print(f'  {i+1}. REF: {refs[i]}', flush=True)
            print(f'     OUT: {preds[i]}', flush=True)
    return exact, chrf, preds

history = []
_, _, preds = report(0)
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=4e-4)
max_steps = 160
max_seconds = 600  # Stops before starting any further optimizer step beyond this limit.
start_time = time.monotonic()
completed = 0

for step in range(1, max_steps+1):
    if time.monotonic() - start_time >= max_seconds:
        print('TRAINING TIME LIMIT REACHED', flush=True)
        break
    model.train()
    model.config.use_cache = False
    opt.zero_grad(set_to_none=True)
    loss = model(**src, labels=targets).loss
    if not torch.isfinite(loss):
        print('NON-FINITE LOSS: stopping', flush=True)
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
    opt.step()
    completed = step
    if step == 1 or step % 20 == 0:
        print(f'Step {step}/{max_steps} loss={loss.item():.4f} elapsed={(time.monotonic()-start_time)/60:.1f}min', flush=True)
    if step % 40 == 0:
        exact, _, preds = report(step)
        if exact == 8:
            print('ALL EIGHT MEMORIZED; STOPPING EARLY', flush=True)
            break

exact, chrf, preds = report(completed)
with (OUT / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tifinagh', 'arabic_reference', 'memorization_prediction'])
    writer.writerows((row['target'], row['source'], pred) for row, pred in zip(examples, preds))
(OUT / 'results.json').write_text(json.dumps({
    'steps': completed,
    'training_minutes': round((time.monotonic()-start_time)/60, 2),
    'history': history,
}, indent=2), encoding='utf-8')
print('\n=== MEMORIZATION RESULT ===', flush=True)
print('Optimizer steps:', completed, flush=True)
print('Exact:', f'{exact}/8', flush=True)
print('chrF:', f'{chrf:.2f}', flush=True)
print('Result CSV:', OUT / 'predictions.csv', flush=True)
print('Existing adapters unchanged.', flush=True)

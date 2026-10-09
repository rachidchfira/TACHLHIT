import csv, json, random, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path.home() / 'soussvoice'
DATA = ROOT / 'datasets/reverse_translation/v2'
OUT = ROOT / 'experiments/language-adaptation/byt5_pilot'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def load(name):
    with (DATA / f'{name}.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get('shi_latn', '').strip() and r.get('en', '').strip()]

train = load('train')
valid = load('validation')
random.shuffle(train)
random.shuffle(valid)
train = train[:64]
valid = valid[:16]
print('Pilot examples:', len(train), 'train;', len(valid), 'validation', flush=True)

model_id = 'google/byt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

PREFIX = 'Translate Tachelhit to English: '

def encode(row):
    x = tokenizer(PREFIX + row['shi_latn'], max_length=160, truncation=True, return_tensors='pt')
    y = tokenizer(text_target=row['en'], max_length=128, truncation=True, return_tensors='pt')
    return {k: v.to(device) for k,v in x.items()}, y['input_ids'].to(device)

@torch.no_grad()
def eval_loss(rows):
    model.eval()
    total = 0.0
    for row in rows:
        inputs, labels = encode(row)
        total += float(model(**inputs, labels=labels).loss.detach().cpu())
    return total / len(rows)

print('Device:', device, flush=True)
start = time.monotonic()
initial = eval_loss(valid)
print(f'Initial validation loss: {initial:.4f}', flush=True)
model.train()
losses=[]
for step, row in enumerate(train, 1):
    inputs, labels = encode(row)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    if step % 8 == 0:
        print(f'Step {step}/64 | mean train loss (last 8): {sum(losses[-8:])/8:.4f} | elapsed: {time.monotonic()-start:.1f}s', flush=True)

final = eval_loss(valid)
print(f'Final validation loss: {final:.4f}', flush=True)
print(f'Elapsed seconds: {time.monotonic()-start:.1f}', flush=True)
report={'model':model_id,'device':str(device),'train_examples':len(train),'validation_examples':len(valid),'initial_validation_loss':initial,'final_validation_loss':final,'steps':len(train),'elapsed_seconds':time.monotonic()-start,'note':'Pilot only; not a reliable quality benchmark.'}
(OUT/'pilot_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Report:',OUT/'pilot_report.json',flush=True)

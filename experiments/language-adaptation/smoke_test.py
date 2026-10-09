import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = 'google/byt5-small'
print('Loading', MODEL, flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)
model.train()
source = 'translate Tachelhit to English: Azul, manik s tgit?'
target = 'Hello, how are you?'
inputs = tokenizer(source, return_tensors='pt', max_length=128, truncation=True)
labels = tokenizer(text_target=target, return_tensors='pt', max_length=96, truncation=True)['input_ids']
labels[labels == tokenizer.pad_token_id] = -100
inputs = {k: v.to(device) for k, v in inputs.items()}
start = time.perf_counter()
outputs = model(**inputs, labels=labels.to(device))
print('Forward loss:', float(outputs.loss.detach().cpu()), flush=True)
outputs.loss.backward()
print('Backward pass: OK', flush=True)
print('Device:', device, 'Elapsed seconds:', round(time.perf_counter()-start, 2), flush=True)
if device.type == 'mps':
    print('MPS allocated GB:', round(torch.mps.current_allocated_memory()/1e9, 2), flush=True)

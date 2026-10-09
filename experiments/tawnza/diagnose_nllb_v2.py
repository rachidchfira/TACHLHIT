"""SoussVoice NLLB V2 diagnostic: seen vs unseen sentences; no training."""
import csv
import random
from pathlib import Path
import torch
import pyarrow.parquet as pq
import sacrebleu
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path('datasets/external/tawnza/clean_v2')
ADAPTER = Path('experiments/tawnza/lora_pilot_v2/adapter')
OUT = Path('experiments/tawnza/lora_pilot_v2/diagnostic_seen_unseen.csv')
BASE = 'facebook/nllb-200-distilled-600M'
assert (ADAPTER / 'adapter_config.json').is_file(), f'Adapter not found: {ADAPTER}'

train = pq.read_table(ROOT / 'train.parquet', columns=['source', 'target']).to_pylist()
random.Random(2026).shuffle(train)
seen = train[:1600]  # matches V2 training order
validation = pq.read_table(ROOT / 'validation.parquet', columns=['source', 'target']).to_pylist()

sets = {
    'seen_in_v2_training': random.Random(11).sample(seen, 40),
    'unseen_validation_episodes': random.Random(12).sample(validation, 40),
}

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('Device:', device, flush=True)
tok = AutoTokenizer.from_pretrained(BASE, src_lang='tzm_Tfng', tgt_lang='arb_Arab')
arabic_id = tok.convert_tokens_to_ids('arb_Arab')
assert arabic_id != tok.unk_token_id, 'Arabic language token not recognized'
base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, str(ADAPTER)).to(device).eval()

all_rows = []
for label, examples in sets.items():
    predictions = []
    for i in range(0, len(examples), 4):
        batch = examples[i:i+4]
        x = tok([r['target'] for r in batch], return_tensors='pt', padding=True,
                truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            gen = model.generate(**x, forced_bos_token_id=arabic_id,
                                 max_new_tokens=80, num_beams=3)
        predictions.extend(tok.batch_decode(gen, skip_special_tokens=True))
    references = [r['source'] for r in examples]
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    exact = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    print(f'{label}: chrF={chrf:.2f}, exact={exact}/{len(examples)}', flush=True)
    for row, predicted in zip(examples, predictions):
        all_rows.append([label, row['target'], row['source'], predicted])
    print('  Example reference:', references[0])
    print('  Example prediction:', predictions[0], flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['subset', 'tifinagh', 'arabic_reference', 'prediction'])
    w.writerows(all_rows)
print('Saved:', OUT)
print('No models or datasets were modified.')

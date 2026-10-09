import csv
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = [
    "Helsinki-NLP/opus-mt-ber-en",
    "Helsinki-NLP/opus-mt-mul-en",
]

PAIRS = [
    ("Manmk?", "How?"),
    ("Manmk rad k sniriɣ?", "How may I serve you?"),
    ("Ftu ad tɣrt.", "Go study."),
    ("Ɣẓannt.", "They're right."),
    ("Ur k sul kṣuḍɣ.", "I'm not scared of you anymore."),
    ("Ḥmlɣ tudrt n tɣrmt.", "I like city life."),
    ("Ack d rkz!", "Come dance!"),
    ("Ur skirɣ amya.", "I did nothing."),
    ("Iɣra.", "He called."),
    ("Ṭṭṣn.", "They slept."),
    ("Rad tccm.", "You will eat."),
    ("Riɣ ad swuriɣ ɣid.", "I want to work here."),
]

OUT = Path(
    "experiments/language-adaptation/marian_benchmark"
)
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device, flush=True)

results = []

for model_id in MODELS:
    print("\n" + "=" * 65)
    print("MODEL:", model_id, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id
    ).to(device)
    model.eval()

    unknown = 0
    total = 0
    exact = 0
    start = time.time()

    for shi, ref in PAIRS:
        encoded = tokenizer(
            shi,
            return_tensors="pt",
            truncation=True,
            max_length=160,
        )

        ids = encoded["input_ids"][0].tolist()
        unk_count = ids.count(tokenizer.unk_token_id)

        unknown += unk_count
        total += len(ids)

        with torch.inference_mode():
            output = model.generate(
                **encoded.to(device),
                num_beams=4,
                max_new_tokens=96,
            )

        prediction = tokenizer.decode(
            output[0], skip_special_tokens=True
        ).strip()

        match = prediction.casefold() == ref.casefold()
        exact += int(match)

        results.append({
            "model": model_id,
            "shi_latn": shi,
            "reference": ref,
            "prediction": prediction,
            "exact": match,
            "unknown_tokens": unk_count,
            "source_tokens": len(ids),
        })

        print(f"\nSHI: {shi}")
        print(f"REF: {ref}")
        print(f"OUT: {prediction}")
        print(f"UNK: {unk_count}/{len(ids)}")
        print(f"EXACT: {match}", flush=True)

    print("\nMODEL SUMMARY")
    print(f"Exact matches: {exact}/{len(PAIRS)}")
    print(f"Unknown tokens: {unknown}/{total}")
    print(f"Elapsed: {time.time()-start:.1f}s")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

with (OUT / "results.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", OUT / "results.csv")

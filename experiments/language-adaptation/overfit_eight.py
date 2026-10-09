import csv
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "google/byt5-small"
MAX_STEPS = 500
LR = 1e-4

OUT = Path(
    "experiments/language-adaptation/byt5_runs/overfit8"
)
OUT.mkdir(parents=True, exist_ok=True)

# Known pairs from our Tachelhit dataset.
PAIRS = [
    ("Manmk?", "How?"),
    ("Manmk rad k sniriɣ?", "How may I serve you?"),
    ("Ftu ad tɣrt.", "Go study."),
    ("Ɣẓannt.", "They're right."),
    ("Ur k sul kṣuḍɣ.", "I'm not scared of you anymore."),
    ("Ḥmlɣ tudrt n tɣrmt.", "I like city life."),
    ("Ack d rkz!", "Come dance!"),
    ("Ur skirɣ amya.", "I did nothing."),
]

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR
)

sources = [
    "Translate Tachelhit to English: " + shi
    for shi, en in PAIRS
]
targets = [en for shi, en in PAIRS]

encoded = tokenizer(
    sources,
    padding=True,
    truncation=True,
    max_length=160,
    return_tensors="pt",
)

target_encoded = tokenizer(
    text_target=targets,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

labels = target_encoded["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

batch = {
    "input_ids": encoded["input_ids"].to(device),
    "attention_mask": encoded["attention_mask"].to(device),
    "labels": labels.to(device),
}

def evaluate(step):
    model.eval()
    correct = 0
    results = []

    with torch.no_grad():
        for shi, ref in PAIRS:
            x = tokenizer(
                "Translate Tachelhit to English: " + shi,
                return_tensors="pt",
            ).to(device)

            ids = model.generate(
                **x,
                max_new_tokens=96,
                num_beams=1,
                do_sample=False,
            )

            pred = tokenizer.decode(
                ids[0], skip_special_tokens=True
            ).strip()

            match = pred.casefold() == ref.casefold()
            correct += int(match)
            results.append((shi, ref, pred, match))

    path = OUT / f"predictions_step_{step}.csv"

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["shi_latn", "reference", "prediction", "exact"]
        )
        writer.writerows(results)

    print(
        f"\nSTEP {step}: {correct}/8 exact",
        flush=True,
    )

    for shi, ref, pred, match in results:
        print(f"SHI: {shi}")
        print(f"REF: {ref}")
        print(f"OUT: {pred}")
        print(f"CORRECT: {match}")
        print("---")

    return correct

start = time.time()

for step in range(1, MAX_STEPS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = model(**batch).loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    optimizer.step()

    if step % 25 == 0:
        print(
            f"Step {step}/{MAX_STEPS} "
            f"| loss {loss.item():.4f} "
            f"| elapsed {(time.time()-start)/60:.1f} min",
            flush=True,
        )

    if step % 50 == 0:
        correct = evaluate(step)

        if correct == 8:
            print(
                "\nSUCCESS: 8/8 translations memorized!",
                flush=True,
            )
            break

print("\nExperiment complete.")
print("Results saved to:", OUT)

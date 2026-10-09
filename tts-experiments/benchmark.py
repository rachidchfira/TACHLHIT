import csv
import re
import torch
from pathlib import Path
from scipy.io.wavfile import write
from transformers import AutoTokenizer, VitsModel

MODEL = "facebook/mms-tts-shi"

sentences = [
    "Azul.",
    "Azul fellawn.",
    "Manik a nn tgit?",
    "Is tgit biḫir?",
    "Nkki labas, tanmmirt.",
    "Mad tskart ass-a?",
    "Mani tllit?",
    "Mani tzzriḍ?",
    "Maḫ tgit akka?",
    "Tanmmirt aṭṭas.",
    "Ar tufat.",
    "Ar timlilit.",
    "Is tssnt tašlḥit?",
    "Nkkni nsawal s tašlḥit.",
    "Riġ ad sawlġ akidk.",
    "Maḫ tḍṣat?",
    "Iḫṣṣa yi ad raḥġ.",
    "Ur ssinġ ma ad skṛġ.",
    "Ma ism nnk?",
    "Ism inu Rachid.",
    "Mani tzdġt?",
    "Nkkni nzdġ ġ Sus.",
    "Ayyuz!",
    "Iġ tṛit, ad nmun.",
    "Ma tṛit ad tssut?",
    "Tga tafukt.",
    "Tafukt tffġd.",
    "Aman iġudan.",
    "Ḫir ad tgit biḫir.",
    "Tanmmirt, ar timlilit.",
]

out = Path("benchmark_audio")
out.mkdir(exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = VitsModel.from_pretrained(MODEL)
model.eval()

vocab = tokenizer.get_vocab()
rows = []

for i, sentence in enumerate(sentences, 1):
    chars = sorted(set(sentence.lower()))
    unsupported = [
        c for c in chars
        if c not in vocab and not c.isspace()
        and c not in ".,?!"
    ]

    filename = f"{i:02d}.wav"

    if unsupported:
        print(f"WARNING {i:02d}: unsupported {unsupported}")

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        waveform = model(
            **inputs
        ).waveform.squeeze().cpu().numpy()

    write(
        str(out / filename),
        model.config.sampling_rate,
        waveform
    )

    rows.append({
        "id": i,
        "sentence": sentence,
        "audio": filename,
        "unsupported_characters": " ".join(unsupported),
        "pronunciation_score_1_to_5": "",
        "notes": "",
    })

    print(f"Generated {filename}: {sentence}")

with open(out / "scores.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nDone. Open benchmark_audio to listen.")

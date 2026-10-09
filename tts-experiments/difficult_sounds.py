import torch
from pathlib import Path
from scipy.io.wavfile import write
from transformers import AutoTokenizer, VitsModel

MODEL_ID = "facebook/mms-tts-shi"

tests = [
    ("01_emphatic_z", "Aẓru."),
    ("02_initial_z", "Ẓru."),
    ("03_between_vowels_z", "Aẓa."),
    ("04_plain_z", "Azru."),
    ("05_ayn_original", "Aεrur."),
    ("06_ayn_between_vowels", "Aεa."),
    ("07_ayn_initial", "εAli."),
    ("08_without_ayn", "Arur."),
]

output = Path("difficult_sounds")
output.mkdir(exist_ok=True)

print("Loading Meta Tachelhit model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID)
model.eval()

for filename, sentence in tests:
    print("Generating:", sentence)

    inputs = tokenizer(sentence, return_tensors="pt")

    if inputs["input_ids"].shape[1] < 2:
        print("Skipping invalid input:", sentence)
        continue

    with torch.no_grad():
        audio = model(**inputs).waveform.squeeze().cpu().numpy()

    filepath = output / f"{filename}.wav"
    write(str(filepath), model.config.sampling_rate, audio)

    print("Saved:", filepath)

print("All tests finished!")

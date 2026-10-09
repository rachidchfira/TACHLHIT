from pathlib import Path
import csv
import subprocess
import torch
import librosa
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "facebook/mms-1b-all"
FOLDER = Path("datasets/recordings")
OUTPUT = Path("datasets/transcripts/asr_results.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading Meta MMS-ASR...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

processor.tokenizer.set_target_lang("shi")
model.load_adapter("shi")
model.eval()

files = sorted(
    p for p in FOLDER.iterdir()
    if p.suffix.lower() in {".wav", ".m4a"}
)

# Avoid transcribing the same recording twice
# when both M4A and converted WAV exist.
by_name = {}
for file in files:
    key = file.stem.lower()
    if key not in by_name or file.suffix.lower() == ".wav":
        by_name[key] = file

files = list(by_name.values())

print(f"Found {len(files)} recordings")

results = []

for index, file in enumerate(files, start=1):
    print(f"\n[{index}/{len(files)}] {file.name}")

    audio_file = file

    if file.suffix.lower() == ".m4a":
        audio_file = file.with_suffix(".wav")
        subprocess.run([
            "afconvert",
            str(file),
            str(audio_file),
            "-f", "WAVE",
            "-d", "LEI16@16000",
            "-c", "1"
        ], check=True)

    audio, sr = librosa.load(
        str(audio_file),
        sr=16000,
        mono=True
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted)[0]

    print("ASR heard:", transcription)

    results.append({
        "filename": file.name,
        "transcription": transcription
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "transcription"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nAll recordings processed!")
print("Results saved to:", OUTPUT)

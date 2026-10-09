from pathlib import Path
import subprocess
import numpy as np
import librosa
import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, Wav2Vec2ForCTC

folder = Path("datasets/recordings")
matches = [
    p for p in folder.iterdir()
    if p.stem.lower().replace("_", " ").strip() == "test full"
    and p.suffix.lower() in {".m4a", ".wav"}
]

if not matches:
    raise SystemExit(
        "Cannot find 'test full.m4a' or 'test full.wav' "
        "in datasets/recordings/"
    )

source = next(
    (p for p in matches if p.suffix.lower() == ".wav"),
    matches[0]
)

wav = folder / "test_full_16k.wav"

if source != wav:
    subprocess.run([
        "afconvert", str(source), str(wav),
        "-f", "WAVE", "-d", "LEI16@16000", "-c", "1"
    ], check=True)

audio, sr = librosa.load(str(wav), sr=16000, mono=True)

# Detect pauses of approximately 0.6 seconds or longer.
intervals = librosa.effects.split(
    audio,
    top_db=30,
    frame_length=1024,
    hop_length=256
)

# Merge nearby speech regions into one utterance.
merged = []
max_gap = int(0.6 * sr)

for start, end in intervals:
    if merged and start - merged[-1][1] < max_gap:
        merged[-1][1] = end
    else:
        merged.append([int(start), int(end)])

print("Detected speech segments:", len(merged))

print("Loading Meta MMS-ASR...")
processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
processor.tokenizer.set_target_lang("shi")
model.load_adapter("shi")
model.eval()

output = folder / "test_full_segments"
output.mkdir(exist_ok=True)

for i, (start, end) in enumerate(merged, 1):
    pad = int(0.2 * sr)
    clip = audio[max(0, start-pad):min(len(audio), end+pad)]

    if len(clip) < int(0.3 * sr):
        continue

    write(str(output / f"segment_{i:02}.wav"), sr, clip)

    inputs = processor(
        clip, sampling_rate=sr, return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted)[0]

    print(f"\nSegment {i}:")
    print("Expected: Is tgit biḫir?")
    print("ASR heard:", transcription)

print("\nSegments saved in:", output)

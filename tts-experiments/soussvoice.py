import torch
from pathlib import Path
from scipy.io.wavfile import write
from transformers import AutoTokenizer, VitsModel
from normalizer import normalize

MODEL_ID = "facebook/mms-tts-shi"

print("Loading Meta Tachelhit model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID)
model.eval()

vocab = tokenizer.get_vocab()

while True:
    original = input("\nEnter Tachelhit (or quit): ").strip()

    if original.lower() in {"quit", "exit", "q"}:
        break

    if not original:
        continue

    normalized = normalize(original)

    unsupported = sorted({
        c for c in normalized.lower()
        if c.isalpha() and c not in vocab
    })

    print("Original:  ", original)
    print("Normalized:", normalized)

    if unsupported:
        print("WARNING: Unsupported characters:", unsupported)
        print("Please correct the spelling before synthesis.")
        continue

    inputs = tokenizer(normalized, return_tensors="pt")

    with torch.no_grad():
        audio = model(**inputs).waveform.squeeze().cpu().numpy()

    filename = Path("soussvoice_output.wav")
    write(str(filename), model.config.sampling_rate, audio)

    print("Saved:", filename.resolve())

    # macOS audio playback
    import subprocess
    subprocess.run(["afplay", str(filename)])

from pathlib import Path
import torch
import librosa
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "facebook/mms-1b-all"
AUDIO_FILE = Path("datasets/recordings/test01.wav")

if not AUDIO_FILE.exists():
    raise SystemExit(f"Recording not found: {AUDIO_FILE}")

print("Loading Meta MMS-ASR...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Tachelhit language adapter
processor.tokenizer.set_target_lang("shi")
model.load_adapter("shi")
model.eval()

audio, sr = librosa.load(str(AUDIO_FILE), sr=16000, mono=True)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(inputs.input_values).logits

tokens = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(tokens)[0]

print("\nExpected: Ism inu Rašid.")
print("ASR heard:", transcription)

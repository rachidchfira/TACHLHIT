import torch
from transformers import AutoTokenizer, VitsModel
from scipy.io.wavfile import write

model_id = "facebook/mms-tts-shi"

print("Downloading Meta Tachelhit model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)
model.eval()

text = "Ism inu Rašid."

inputs = tokenizer(text, return_tensors="pt")

print("Generating speech...")
with torch.no_grad():
    audio = model(**inputs).waveform.squeeze().cpu().numpy()

write("meta_tachelhit.wav", model.config.sampling_rate, audio)

print("Done! Opening audio...")

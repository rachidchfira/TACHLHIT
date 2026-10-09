from pathlib import Path
import librosa
import soundfile as sf

source = Path("datasets/recordings/test_full_16k.wav")
output = Path("datasets/recordings/repetition_checks")
output.mkdir(exist_ok=True)

audio, sr = librosa.load(str(source), sr=16000, mono=True)

duration = len(audio) / sr
print(f"Recording duration: {duration:.2f} seconds")

for i in range(3):
    start = int(i * len(audio) / 3)
    end = int((i + 1) * len(audio) / 3)

    clip = audio[start:end]
    path = output / f"part_{i+1}.wav"
    sf.write(path, clip, sr)

    print(f"Part {i+1}: {path}")

print("Listen to each part before evaluating it.")

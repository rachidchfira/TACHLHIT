import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "google/byt5-small"
SOURCE = "Translate Tachelhit to English: Manmk?"
TARGET = "How?"
STEPS = 400

torch.manual_seed(42)
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

inputs = tok(SOURCE, return_tensors="pt").to(device)
labels = tok(text_target=TARGET, return_tensors="pt")[
    "input_ids"
].to(device)

print("Device:", device)
print("Target IDs:", labels[0].tolist(), flush=True)

for step in range(1, STEPS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = model(**inputs, labels=labels).loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 25 == 0:
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
            )
            prediction = tok.decode(
                generated[0], skip_special_tokens=True
            )

            outputs = model(**inputs, labels=labels)
            predicted_ids = outputs.logits.argmax(-1)[0].tolist()
            teacher_prediction = tok.decode(
                predicted_ids, skip_special_tokens=True
            )

        print(
            f"Step {step} | loss={loss.item():.4f}"
            f" | generated={prediction!r}"
            f" | teacher_forced={teacher_prediction!r}",
            flush=True,
        )

        if prediction.strip() == TARGET:
            print("SUCCESS: memorized one sentence!", flush=True)
            break

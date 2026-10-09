import csv
import json
import urllib.request
from pathlib import Path

INPUT = Path("datasets/evaluation/native_benchmark.csv")
OUTPUT = Path("datasets/evaluation/qwen_baseline.csv")

SYSTEM_PROMPT = """
You are SoussVoice, a conversational assistant
speaking everyday Souss/Agadir Tachelhit.

Understand the user's Tachelhit and respond naturally
in Tachelhit using Latin characters.

Do not translate unless requested.
Do not invent meanings.
Keep responses concise.
"""

results = []

with INPUT.open(encoding="utf-8", newline="") as f:
    examples = list(csv.DictReader(f))

for example in examples:
    question = example["question_latin"]

    payload = {
        "model": "qwen3:4b",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(
            request, timeout=180
        ) as response:
            data = json.load(response)

        answer = data.get("message", {}).get("content", "")

    except Exception as e:
        answer = f"ERROR: {e}"

    print(f"\nQuestion: {question}")
    print(f"Qwen: {answer}")

    results.append({
        "id": example["id"],
        "question": question,
        "model_response": answer,
        "native_score": "",
        "native_notes": ""
    })

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id", "question", "model_response",
            "native_score", "native_notes"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to {OUTPUT}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mms-tts-shi"
)

words = ["bikhir", "bixir", "khir", "b", "kh", "x"]

for word in words:
    tokens = tokenizer.tokenize(word)
    ids = tokenizer(word, add_special_tokens=False).input_ids
    print(f"{word:10} tokens={tokens} ids={ids}")

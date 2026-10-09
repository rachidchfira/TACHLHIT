from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mms-tts-shi"
)

print("Tokenizer type:", type(tokenizer).__name__)
print("Vocabulary size:", len(tokenizer))
print("Unknown token:", repr(tokenizer.unk_token))
print("Unknown token ID:", tokenizer.unk_token_id)

print("\nVocabulary:")
for token, idx in sorted(
    tokenizer.get_vocab().items(),
    key=lambda item: item[1]
):
    print(idx, repr(token))

print("\nCharacter tests:")
for char in ["b", "k", "h", "x", "ɣ", "ḥ", "ⵅ", "خ"]:
    print(
        repr(char),
        "->",
        tokenizer(char, add_special_tokens=False).input_ids
    )

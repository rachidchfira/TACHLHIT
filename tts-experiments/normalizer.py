import re
import unicodedata

# Confirmed pronunciation corrections
WORD_OVERRIDES = {
    "rachid": "Rašid",
    "bikhir": "biḫir",
}

# Candidate spelling conventions.
# Add rules only after testing them with native speakers.
RULES = [
    ("kh", "ḫ"),
    ("ch", "š"),
    ("sh", "š"),
]

def normalize(text):
    text = unicodedata.normalize("NFC", text)

    def convert_word(match):
        word = match.group(0)
        key = word.lower()

        if key in WORD_OVERRIDES:
            result = WORD_OVERRIDES[key]
        else:
            result = key
            for original, replacement in RULES:
                result = result.replace(original, replacement)

        if word.isupper():
            return result.upper()
        if word[0].isupper():
            return result[0].upper() + result[1:]
        return result[0].lower() + result[1:]

    return re.sub(r"[^\W\d_]+", convert_word, text, flags=re.UNICODE)

if __name__ == "__main__":
    examples = [
        "Ism inu Rachid.",
        "Is tgit bikhir?",
        "Azul Rachid, manik a nn tgit?",
    ]

    for example in examples:
        print("Original:  ", example)
        print("Normalized:", normalize(example))
        print()

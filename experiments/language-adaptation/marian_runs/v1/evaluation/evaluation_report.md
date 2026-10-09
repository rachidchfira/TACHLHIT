# SoussVoice Marian V1 Evaluation

Test examples: 1103

| Model | BLEU | chrF |
|---|---:|---:|
| Original Marian | 1.7842 | 16.0136 |
| Fine-tuned Marian | 16.5562 | 31.425 |

## Runtime
- Original Marian: 156.9 seconds
- Fine-tuned Marian: 65.9 seconds

## Interpretation
BLEU and chrF measure similarity to reference translations.
They do not establish semantic correctness or dialect fidelity.

Native-speaker review is required before using these predictions
as verified training labels.

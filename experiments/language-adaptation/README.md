# SoussVoice reverse translation dataset v1

This script prepares **Tachelhit Latin -> English** parallel text from the original Tatoeba English-aligned config, not from the previously merged unreviewed corpus.

## Run on the Mac

```bash
cd ~/soussvoice
source services/asr/.venv/bin/activate
python -m pip show datasets
python experiments/language-adaptation/prepare_reverse_dataset.py
```

Copy `prepare_reverse_dataset.py` into `~/soussvoice/experiments/language-adaptation/` first.

## Outputs

`datasets/reverse_translation/v1/{all,train,validation,test,rejected,conflicts}.csv` and `report.json`.

## Important limitations

- This is a **data preparation pass**, not training or proof of model quality.
- Conflicting English translations for the same normalized Tachelhit sentence are quarantined for review.
- Duplicate normalized Tachelhit inputs are assigned to only one split.
- Near-duplicate sentences, dataset contamination and dialect quality require additional audits before claiming trustworthy evaluation.
- License and attribution must be reviewed before redistributing the data or model.

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "Helsinki-NLP/opus-mt-ber-en"
DATA_DIR = Path("datasets/reverse_translation/v2")
OUT = Path("experiments/language-adaptation/marian_runs/v1")

class ParallelDataset(Dataset):
    def __init__(self, path, limit=None):
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [
            (r["shi_latn"].strip(), r["en"].strip())
            for r in rows
            if r.get("shi_latn", "").strip() and r.get("en", "").strip()
        ]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)

    train_data = ParallelDataset(DATA_DIR / "train.csv", args.train_limit)
    val_data = ParallelDataset(DATA_DIR / "validation.csv", args.val_limit)

    def collate(batch):
        sources, targets = zip(*batch)
        inputs = tokenizer(
            list(sources),
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=list(targets),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val = float("inf")
    checkpoint = OUT / "latest.pt"

    if args.resume and checkpoint.exists():
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best_val = state["best_val"]
        print("Resumed from epoch", start_epoch, flush=True)

    print("Training pairs:", len(train_data))
    print("Validation pairs:", len(val_data))
    print("Steps per epoch:", len(train_loader), flush=True)

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        model.train()
        total_train = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += loss.item()

            if step % 25 == 0:
                print(
                    f"Epoch {epoch+1} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        total_val = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                total_val += model(**batch).loss.item()

        train_loss = total_train / len(train_loader)
        val_loss = total_val / len(val_loader)

        print(
            f"\nEPOCH {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"elapsed={time.time()-start:.1f}s",
            flush=True,
        )

        # Save model and optimizer for resuming.
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": min(best_val, val_loss),
        }, checkpoint)

        if val_loss < best_val:
            best_val = val_loss
            best_dir = OUT / "best_model"
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            print("Saved new best model:", best_dir, flush=True)

        # Show translations from held-out validation examples.
        print("\nVALIDATION TRANSLATIONS:")
        for source, reference in val_data.rows[:8]:
            inputs = tokenizer(
                source, return_tensors="pt", truncation=True,
                max_length=160
            ).to(device)

            with torch.inference_mode():
                output = model.generate(
                    **inputs, num_beams=4, max_new_tokens=80
                )

            prediction = tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            print("SHI:", source)
            print("REF:", reference)
            print("OUT:", prediction)
            print("---", flush=True)

    print("\nExperiment complete.")
    print("Best validation loss:", best_val)
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()

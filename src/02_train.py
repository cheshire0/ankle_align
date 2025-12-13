#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_train.py (CPU-friendly baseline)

Input:
- /data/processed/metadata.csv created by 01_data_processing.py
  Columns used: split, label_id, out_path (or source_path fallback)

Output (saved to /app/output):
- model_best.pt           (best validation accuracy)
- train_history.csv       (epoch metrics)
- train_config.json       (settings)
- label_map.json          (id -> label name)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = Path("/data/processed")
META_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = Path("/app/output")

# Must match 01_data_processing.py mapping
ID_TO_LABEL = {
    0: "1_Pronacio",
    1: "2_Neutralis",
    2: "3_Szupinacio",
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class AnkleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_img(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        # Resize and convert to tensor in [0,1], CHW
        img = img.resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        # Prefer processed out_path
        path = row.get("out_path")
        if not isinstance(path, str) or not path:
            path = row["source_path"]

        x = self._preprocess(self._load_img(path))
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows/CPU for stability.")
    parser.add_argument("--use-class-weights", action="store_true", help="Helps imbalance; good for incremental.")
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {META_PATH}. Run 01_data_processing.py first.")

    df = pd.read_csv(META_PATH)

    # Keep only rows that have out_path existing (processed images)
    df = df[df["out_path"].notna()].copy()
    df = df[df["out_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Train/val split is empty. Check metadata.csv and processed images.")

    device = torch.device("cpu")

    train_ds = AnkleDataset(train_df, img_size=args.img_size)
    val_ds = AnkleDataset(val_df, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallCNN(num_classes=3).to(device)

    # Loss
    if args.use_class_weights:
        counts = train_df["label_id"].value_counts().to_dict()
        w = []
        for k in range(3):
            w.append(1.0 / max(counts.get(k, 1), 1))
        w = torch.tensor(w, dtype=torch.float32)
        w = w / w.sum() * 3.0
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(logits, yb) * bs
            n += bs
            pbar.set_postfix(loss=running_loss / max(n, 1), acc=running_acc / max(n, 1))

        train_metrics = {"loss": running_loss / max(n, 1), "acc": running_acc / max(n, 1)}
        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        print(f"Epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} "
              f"| val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "img_size": args.img_size,
                    "id_to_label": ID_TO_LABEL,
                },
                OUTPUT_DIR / "model_best.pt",
            )

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "train_history.csv", index=False)

    with open(OUTPUT_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in ID_TO_LABEL.items()}, f, indent=2)

    print(f"Done. Best val acc: {best_val_acc:.4f}")
    print(f"Saved: {OUTPUT_DIR / 'model_best.pt'}")


if __name__ == "__main__":
    main()

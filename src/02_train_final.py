#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_train_final.py

Final model trainer:
- Small CNN (2 conv blocks + FC)
- CPU-only
- NO augmentation
- NO class weights
- NO early stopping (fixed epochs)

Inputs:
- /data/processed/metadata.csv  (from 01_data_processing.py)

Outputs (/app/output):
- model_final.pt
- train_history_final.csv
- train_config_final.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import setup_logger

logger = setup_logger()

# Pillow tolerance for truncated JPEGs (does not rotate or fix EXIF)
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = Path("/data/processed")
META_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = Path("/app/output")

ID_TO_LABEL = {
    0: "1_Pronacio",
    1: "2_Neutralis",
    2: "3_Szupinacio",
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    return torch.from_numpy(arr)


class FinalDataset(Dataset):
    """Final model dataset: NO augmentation, resize only."""
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = row["out_path"]
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)

        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        x = to_tensor(img)
        return x, y


class TinyCNNFinal(nn.Module):
    """
    Final model:
      Conv(3->16) + pool
      Conv(16->32) + pool
      Flatten + FC
    """
    def __init__(self, num_classes: int = 3, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
        )

        feat_size = img_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * feat_size * feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
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
        total_acc += accuracy_from_logits(logits, yb) * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting FINAL training (TinyCNN: 2 conv blocks + FC)")
    logger.info("Final configuration:")
    logger.info(f"  img_size: {args.img_size}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  lr: {args.lr}")
    logger.info(f"  weight_decay: {args.weight_decay}")
    logger.info(f"  seed: {args.seed}")
    logger.info(f"  num_workers: {args.num_workers}")
    logger.info(f"Metadata path: {META_PATH}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {META_PATH}. Run 01_data_processing.py first.")

    df = pd.read_csv(META_PATH)
    logger.info(f"Loaded metadata rows: {len(df)}")

    # Keep only existing processed items
    df = df[df["out_path"].notna()].copy()
    df = df[df["out_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())].copy()
    logger.info(f"Rows with existing processed images: {len(df)}")

    logger.info("Split sizes:")
    for split_name, cnt in df["split"].value_counts().items():
        logger.info(f"  {split_name}: {cnt}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Train/val split is empty. Check metadata.csv and processed images.")

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")

    logger.info("Train label distribution:")
    for label, cnt in train_df["label_id"].value_counts().sort_index().items():
        logger.info(f"  {ID_TO_LABEL.get(int(label), str(label))}: {cnt}")

    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    train_ds = FinalDataset(train_df, img_size=args.img_size)
    val_ds = FinalDataset(val_df, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TinyCNNFinal(num_classes=3, img_size=args.img_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: TinyCNNFinal")
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    criterion = nn.CrossEntropyLoss()  # final model: no class weights
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    history = []

    ckpt_path = OUTPUT_DIR / "model_final.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        run_acc = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Final Epoch {epoch}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            run_loss += loss.item() * bs
            run_acc += accuracy_from_logits(logits, yb) * bs
            n += bs
            pbar.set_postfix(loss=run_loss / max(n, 1), acc=run_acc / max(n, 1))

        train_metrics = {"loss": run_loss / max(n, 1), "acc": run_acc / max(n, 1)}
        val_metrics = evaluate(model, val_loader, device, criterion)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
        )

        logger.info(
            f"Final Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc + 1e-6:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            logger.info(f"New best FINAL val_acc={best_val_acc:.4f} at epoch {epoch}. Saving checkpoint.")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "img_size": args.img_size,
                    "id_to_label": ID_TO_LABEL,
                    "model_name": "TinyCNNFinal",
                },
                ckpt_path,
            )

    hist_path = OUTPUT_DIR / "train_history_final.csv"
    cfg_path = OUTPUT_DIR / "train_config_final.json"

    pd.DataFrame(history).to_csv(hist_path, index=False)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"FINAL training finished. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    logger.info(f"Saved checkpoint: {ckpt_path}")
    logger.info(f"Saved history: {hist_path}")
    logger.info(f"Saved config: {cfg_path}")


if __name__ == "__main__":
    main()

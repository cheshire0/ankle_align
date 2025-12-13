#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_train_v2.py

CPU-friendly incremental trainer:
- GAP CNN (much fewer parameters than FC-heavy baseline)
- NO rotation augmentation (explicitly excluded)
- Early stopping on validation accuracy
- Class weights in CrossEntropyLoss

Input:
- /data/processed/metadata.csv  (from 01_data_processing.py)
  Uses: split, label_id, out_path

Output (/app/output):
- model_best_v2.pt
- train_history_v2.csv
- train_config_v2.json
- label_map.json (if not already there)
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

# Pillow can sometimes choke on truncated JPEGs; this makes it more tolerant.
# (Does NOT rotate. Does NOT "fix" EXIF. Only affects truncated decoding.)
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = Path("/data/processed")
META_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = Path("/app/output")

# Must match your pipeline mapping
ID_TO_LABEL = {
    0: "1_Pronacio",
    1: "2_Neutralis",
    2: "3_Szupinacio",
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -------------------------
# Minimal transforms (NO ROTATION)
# -------------------------
def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    return torch.from_numpy(arr)


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size))


def random_resized_crop_no_rotate(img: Image.Image, size: int, scale=(0.85, 1.0)) -> Image.Image:
    """Random crop+resize (zoom) with NO rotation."""
    w, h = img.size
    area = w * h
    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect = np.random.uniform(0.9, 1.1)
        new_w = int(round(np.sqrt(target_area * aspect)))
        new_h = int(round(np.sqrt(target_area / aspect)))
        if new_w <= w and new_h <= h:
            x1 = np.random.randint(0, w - new_w + 1)
            y1 = np.random.randint(0, h - new_h + 1)
            cropped = img.crop((x1, y1, x1 + new_w, y1 + new_h))
            return cropped.resize((size, size))
    # fallback to center crop-ish
    return img.resize((size, size))


def color_jitter(img: Image.Image, brightness=0.15, contrast=0.15) -> Image.Image:
    """Simple brightness/contrast jitter (NO rotation)."""
    # brightness
    if brightness > 0:
        b = np.random.uniform(1 - brightness, 1 + brightness)
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) * b, 0, 255).astype(np.uint8))
    # contrast
    if contrast > 0:
        c = np.random.uniform(1 - contrast, 1 + contrast)
        arr = np.asarray(img, np.float32)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        arr = np.clip((arr - mean) * c + mean, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def random_translate_no_rotate(img: Image.Image, max_frac: float = 0.06) -> Image.Image:
    """Translation only; NO rotation, NO shear."""
    w, h = img.size
    max_dx = int(round(w * max_frac))
    max_dy = int(round(h * max_frac))
    dx = np.random.randint(-max_dx, max_dx + 1)
    dy = np.random.randint(-max_dy, max_dy + 1)
    # Paste onto black canvas
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img, (dx, dy))
    return canvas


# -------------------------
# Dataset
# -------------------------
class AnkleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = row["out_path"]
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)

        img = Image.open(path).convert("RGB")

        # IMPORTANT: no rotation augmentation. Only crop/resize, jitter, translate.
        if self.train:
            img = random_resized_crop_no_rotate(img, self.img_size, scale=(0.85, 1.0))
            if np.random.rand() < 0.7:
                img = color_jitter(img, brightness=0.15, contrast=0.15)
            if np.random.rand() < 0.5:
                img = random_translate_no_rotate(img, max_frac=0.06)
        else:
            img = resize(img, self.img_size)

        x = to_tensor(img)
        return x, y


# -------------------------
# Model: GAP CNN
# -------------------------
class GAPCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.gap(x)
        x = self.head(x)
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


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["label_id"].value_counts().to_dict()
    w = []
    for k in range(3):
        w.append(1.0 / max(int(counts.get(k, 1)), 1))
    w = torch.tensor(w, dtype=torch.float32)
    # normalize to mean weight = 1
    w = w / w.mean()
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (val acc).")
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {META_PATH}. Run 01_data_processing.py first.")

    df = pd.read_csv(META_PATH)

    # Keep only processed items that exist
    df = df[df["out_path"].notna()].copy()
    df = df[df["out_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Train/val split is empty. Check metadata.csv and processed images.")

    device = torch.device("cpu")

    train_ds = AnkleDataset(train_df, img_size=args.img_size, train=True)
    val_ds = AnkleDataset(val_df, img_size=args.img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GAPCNN(num_classes=3).to(device)

    class_w = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        run_acc = 0.0
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
            run_loss += loss.item() * bs
            run_acc += accuracy_from_logits(logits, yb) * bs
            n += bs
            pbar.set_postfix(loss=run_loss / max(n, 1), acc=run_acc / max(n, 1))

        train_metrics = {"loss": run_loss / max(n, 1), "acc": run_acc / max(n, 1)}
        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

        # Early stopping on val acc
        if val_metrics["acc"] > best_val_acc + 1e-6:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "img_size": args.img_size,
                    "id_to_label": ID_TO_LABEL,
                    "class_weights": class_w.detach().cpu().tolist(),
                },
                OUTPUT_DIR / "model_best_v2.pt",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping: no val acc improvement for {args.patience} epochs. Best epoch: {best_epoch}")
            break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "train_history_v2.csv", index=False)

    with open(OUTPUT_DIR / "train_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Keep label_map.json consistent
    label_map_path = OUTPUT_DIR / "label_map.json"
    if not label_map_path.exists():
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in ID_TO_LABEL.items()}, f, indent=2)

    print(f"Done. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Saved: {OUTPUT_DIR / 'model_best_v2.pt'}")


if __name__ == "__main__":
    main()

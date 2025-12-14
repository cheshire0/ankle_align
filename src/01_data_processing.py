#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_data_processing.py

Reads Label Studio exports stored in per-Neptun folders, finds the corresponding images,
cleans/skips unusable files, resizes, and writes a processed dataset with unique filenames
(neptun-prefixed) to avoid collisions across folders.

Expected input layout (inside Docker):
/data/
  raw/
    <NEPTUN1>/
      ...images... (any subfolders allowed)
      ...label studio export... (*.json)
    <NEPTUN2>/
      ...
  processed/   (will be created/overwritten)

Outputs:
- /data/processed/images/{train,val,test}/{label_name}/<NEPTUN>__<original_name>
- /data/processed/metadata.csv
- /app/output/data_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import setup_logger
logger = setup_logger()

# Inside-container paths (DO NOT change to C:\ paths)
RAW_ROOT = Path("/data/raw")
PROCESSED_ROOT = Path("/data/processed")
OUTPUT_DIR = Path("/app/output")

LABEL_MAP = {
    "1_Pronacio": 0,
    "2_Neutralis": 1,
    "3_Szupinacio": 2,
}
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass
class Item:
    neptun: str
    img_path: Path
    original_name: str
    label_name: str
    label_id: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_json_files(neptun_dir: Path) -> List[Path]:
    # Any *.json under this neptun folder counts; we will parse those that look like Label Studio exports.
    return [p for p in neptun_dir.rglob("*.json") if p.is_file()]


def index_images(neptun_dir: Path) -> Dict[str, List[Path]]:
    """
    Build a mapping from basename -> list of full paths within this neptun dir.
    This handles duplicates inside a folder (rare) and supports nested image dirs.
    """
    idx: Dict[str, List[Path]] = {}
    for p in neptun_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name, []).append(p)
    return idx


def is_label_studio_export(obj) -> bool:
    # Label Studio export is commonly a list of dict items with keys like 'file_upload' and 'annotations'
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))


def extract_choice(entry: dict) -> Optional[str]:
    """
    Extract the single-choice classification label from one Label Studio entry.
    Expected structure per spec:
      entry["annotations"][0]["result"][0]["value"]["choices"][0] -> e.g. "1_Pronacio"
    We search robustly in case of extra nesting or multiple results.
    """
    annotations = entry.get("annotations", [])
    for ann in annotations:
        results = ann.get("result", [])
        for r in results:
            val = r.get("value", {})
            choices = val.get("choices")
            if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], str):
                return choices[0]
    return None


def parse_labelstudio_json(json_path: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (file_upload_basename, label_choice_string).
    Skips entries without usable labels.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not is_label_studio_export(data):
        return []

    out: List[Tuple[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        fname = entry.get("file_upload")
        if not isinstance(fname, str) or not fname.strip():
            continue
        choice = extract_choice(entry)
        if choice is None:
            continue
        out.append((Path(fname).name, choice))
    return out


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        return None


def prepare_output_dirs(clean: bool) -> None:
    if clean and PROCESSED_ROOT.exists():
        shutil.rmtree(PROCESSED_ROOT, ignore_errors=True)
    (PROCESSED_ROOT / "images").mkdir(parents=True, exist_ok=True)
    logger.info(f"Processed dataset root: {PROCESSED_ROOT}")
    logger.info(f"Processed images root: {PROCESSED_ROOT / 'images'}")
    logger.info(f"Output report dir: {OUTPUT_DIR}")


def unique_out_name(neptun: str, original_name: str) -> str:
    # Prevent collisions across folders by prefixing with neptun
    return f"{neptun}__{original_name}"


def save_resized(img: Image.Image, out_path: Path, size: int) -> None:
    img = img.convert("RGB")
    img = img.resize((size, size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=95, optimize=True)


def collect_items(raw_root: Path) -> Tuple[List[Item], List[str]]:
    """
    Walk /data/raw/<NEPTUN>/... and collect labeled images from each Neptun folder.

    Returns: (items, warnings)
    """
    warnings: List[str] = []
    items: List[Item] = []

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW_ROOT not found: {raw_root}")

    neptun_dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    if not neptun_dirs:
        raise RuntimeError(f"No Neptun folders found under {raw_root}. Expected /data/raw/<NEPTUN>/...")

    for neptun_dir in neptun_dirs:
        neptun = neptun_dir.name
        img_index = index_images(neptun_dir)
        json_files = find_json_files(neptun_dir)

        if not json_files:
            warnings.append(f"[{neptun}] No JSON label file found (no *.json). Skipping folder.")
            continue

        # Parse all JSON files; keep the ones that yield label pairs
        pairs: List[Tuple[str, str]] = []
        for jp in json_files:
            pairs.extend(parse_labelstudio_json(jp))

        if not pairs:
            warnings.append(f"[{neptun}] JSON files found but none look like Label Studio exports. Skipping folder.")
            continue

        for base_name, choice in pairs:
            if choice not in LABEL_MAP:
                warnings.append(f"[{neptun}] Unknown label '{choice}' for file '{base_name}'. Skipping item.")
                continue

            candidates = img_index.get(base_name, [])

            # If Label Studio added a prefix like "d1a7dc20-<realname>.jpg", strip it
            if not candidates and "-" in base_name:
                prefix, rest = base_name.split("-", 1)
                # Only strip if prefix looks like a hash (hex) to avoid breaking legit names
                if len(prefix) >= 6 and all(c in "0123456789abcdefABCDEF" for c in prefix):
                    candidates = img_index.get(rest, [])
                    if candidates:
                        base_name = rest  # treat the real filename as the original name

            if not candidates:
                warnings.append(f"[{neptun}] Image '{base_name}' referenced in JSON not found under folder.")
                continue


            # If duplicates exist within the same Neptun folder, pick the first path deterministically
            img_path = sorted(candidates)[0]
            items.append(
                Item(
                    neptun=neptun,
                    img_path=img_path,
                    original_name=base_name,
                    label_name=choice,
                    label_id=LABEL_MAP[choice],
                )
            )

    return items, warnings


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
) -> pd.DataFrame:
    """
    Splits into train/val/test stratified by label_id.

    val_size is fraction of the remaining after test split.
    Example: test=0.15, val=0.15 means:
      test = 15%
      val = 15% of remaining 85% = 12.75% overall
      train = rest
    """
    if df.empty:
        return df

    # First: train+val vs test
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )

    # Second: train vs val (fraction of trainval)
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        stratify=trainval_df["label_id"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out


def write_report(df: pd.DataFrame, warnings: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "data_report.txt"

    lines: List[str] = []
    lines.append("AnkleAlign - Data Processing Report\n")
    lines.append(f"Total labeled items: {len(df)}\n")

    if not df.empty:
        lines.append("Counts by label:\n")
        counts = df["label_name"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"  {k}: {v}\n")

        lines.append("\nCounts by split and label:\n")
        pivot = pd.pivot_table(
            df,
            index=["split"],
            columns=["label_name"],
            values="out_filename",
            aggfunc="count",
            fill_value=0,
        )
        lines.append(pivot.to_string())
        lines.append("\n")

    if warnings:
        lines.append("\nWarnings / Skips:\n")
        for w in warnings[:300]:
            lines.append(f"- {w}\n")
        if len(warnings) > 300:
            lines.append(f"... truncated ({len(warnings) - 300} more)\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    logger.info("Starting data preprocessing")
    logger.info(f"Raw data directory: {RAW_ROOT}")
    logger.info(f"Processed output directory: {PROCESSED_ROOT}")
    logger.info(f"App output directory: {OUTPUT_DIR}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=224, help="Resize images to NxN (default 224).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction (default 0.15).")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split fraction (default 0.15).")
    parser.add_argument("--clean", action="store_true", help="Delete /data/processed before writing.")
    args = parser.parse_args()

    logger.info("Preprocessing configuration:")
    logger.info(f"  img_size: {args.img_size}")
    logger.info(f"  seed: {args.seed}")
    logger.info(f"  test_size: {args.test_size}")
    logger.info(f"  val_size: {args.val_size}")
    logger.info(f"  clean: {args.clean}")

    set_seed(args.seed)
    prepare_output_dirs(clean=args.clean)

    items, warnings = collect_items(RAW_ROOT)

    logger.info(f"Collected labeled items (before dedup): {len(items)}")
    if warnings:
        logger.warning(f"Warnings so far: {len(warnings)} (see report for details)")


    if not items:
        raise RuntimeError(
            "No labeled items collected. Check that:\n"
            "1) /data/raw/<NEPTUN>/ contains images\n"
            "2) /data/raw/<NEPTUN>/ contains Label Studio export JSON\n"
            "3) JSON 'file_upload' matches actual image basenames\n"
        )

    # Build dataframe
    df = pd.DataFrame(
        [
            {
                "neptun": it.neptun,
                "img_path": str(it.img_path),
                "original_name": it.original_name,
                "label_name": it.label_name,
                "label_id": it.label_id,
            }
            for it in items
        ]
    )

    # Remove duplicates (same neptun + same filename) keeping first
    df = df.drop_duplicates(subset=["neptun", "original_name"], keep="first").reset_index(drop=True)

    logger.info(f"Items after deduplication: {len(df)}")

    logger.info("Label distribution (after dedup):")
    for label, cnt in df["label_name"].value_counts().items():
        logger.info(f"  {label}: {cnt}")

    # Stratified split
    df = stratified_split(df, seed=args.seed, test_size=args.test_size, val_size=args.val_size)

    logger.info("Split distribution:")
    split_counts = df["split"].value_counts().to_dict()
    for k in ["train", "val", "test"]:
        logger.info(f"  {k}: {split_counts.get(k, 0)}")

    logger.info("Split x label distribution:")
    pivot = pd.pivot_table(
        df,
        index=["split"],
        columns=["label_name"],
        values="original_name",
        aggfunc="count",
        fill_value=0,
    )
    for line in pivot.to_string().splitlines():
        logger.info(line)

    # Process and write images
    out_records: List[dict] = []
    images_root = PROCESSED_ROOT / "images"

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        neptun = row["neptun"]
        img_path = Path(row["img_path"])
        split = row["split"]
        label_name = row["label_name"]
        original_name = row["original_name"]

        img = safe_open_image(img_path)
        if img is None:
            warnings.append(f"[{neptun}] Unreadable image: {img_path}")
            continue

        out_name = unique_out_name(neptun, original_name)
        out_name = Path(out_name).with_suffix(".jpg").name  # normalize output to jpg
        out_path = images_root / split / label_name / out_name

        try:
            save_resized(img, out_path, size=args.img_size)
        except Exception as e:
            warnings.append(f"[{neptun}] Failed to save processed image '{img_path}': {e}")
            continue

        out_records.append(
            {
                "neptun": neptun,
                "split": split,
                "label_name": label_name,
                "label_id": int(row["label_id"]),
                "original_name": original_name,
                "source_path": str(img_path),
                "out_path": str(out_path),
                "out_filename": out_name,
            }
        )

    out_df = pd.DataFrame(out_records)
    meta_path = PROCESSED_ROOT / "metadata.csv"
    out_df.to_csv(meta_path, index=False, encoding="utf-8")

    logger.info(f"Processed images successfully: {len(out_df)}")
    logger.info(f"Skipped/failed items during processing: {len(df) - len(out_df)}")

    if warnings:
        logger.warning(f"Total warnings: {len(warnings)}")
        # Show first few in log (full list already saved in report)
        for w in warnings[:20]:
            logger.warning(w)
        if len(warnings) > 20:
            logger.warning(f"... warnings truncated in log ({len(warnings)-20} more; see data_report.txt)")

    write_report(out_df, warnings)

    logger.info(f"Done. Processed: {len(out_df)}/{len(df)}")
    logger.info(f"Metadata written: {meta_path}")
    logger.info(f"Report written: {OUTPUT_DIR / 'data_report.txt'}")


if __name__ == "__main__":
    main()

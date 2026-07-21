"""Scan an image-folder dataset (class-subfolder layout) into a table.

Layout expected (Kaggle mount, read-only):
    <train_root>/<class_name>/<image files>
Produces a DataFrame of (image_path, label) where image_path is POSIX-relative
to train_root so the index files stay portable across mount points / machines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp"}


def scan_dataset(
    root: str | Path,
    classes: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """List all images under `root`, one row per image.

    Args:
        root: dataset root containing one subfolder per class.
        classes: expected class names. If given, only these folders are read and
            missing/extra folders are reported. If None, folders are discovered.

    Returns:
        (df, info) where df has columns [image_path, label] and info records the
        discovered subdirs plus any mismatch against `classes`.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    subdirs = sorted(d.name for d in root.iterdir() if d.is_dir())

    if classes:
        missing = [c for c in classes if c not in subdirs]
        extra = [d for d in subdirs if d not in classes]
        use = list(classes)
    else:
        missing, extra = [], []
        use = subdirs

    rows: list[tuple[str, str]] = []
    for cls in use:
        cdir = root / cls
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                # store POSIX-relative path so CSV indices are portable
                rows.append((f"{cls}/{f.name}", cls))

    df = pd.DataFrame(rows, columns=["image_path", "label"])
    info = {
        "root": str(root),
        "subdirs": subdirs,
        "missing_classes": missing,
        "extra_folders": extra,
        "num_images": len(df),
    }
    return df, info

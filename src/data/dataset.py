"""Torch Dataset + transforms reading from the cached split index files.

Augmentation is identical across model families (CLAUDE.md constraint #2) and
deliberately excludes horizontal flip (constraint / pitfall: some letters are
orientation-sensitive). This module imports torch/torchvision; keep the rest of
the data pipeline (scan/dedup/split) torch-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_class_to_idx(classes: list[str]) -> dict[str, int]:
    """Stable class->index map (sorted for reproducibility)."""
    return {c: i for i, c in enumerate(sorted(classes))}


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    """Build train or eval transforms from the data config.

    Train: RandomResizedCrop -> rotation -> color jitter -> RandAugment -> norm.
    Eval:  Resize(shorter side) -> CenterCrop -> norm.
    No horizontal flip in either path.
    """
    size = int(cfg["image"]["size"])
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if not train:
        resize = int(round(size * 256 / 224))  # standard 256->224 style ratio
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

    aug = cfg["augmentation"]
    scale = tuple(aug["random_resized_crop"]["scale"])
    cj = aug["color_jitter"]
    ra = aug["rand_augment"]

    ops: list = [
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomRotation(float(aug.get("rotation_degrees", 0))),
        transforms.ColorJitter(
            brightness=cj.get("brightness", 0.0),
            contrast=cj.get("contrast", 0.0),
            saturation=cj.get("saturation", 0.0),
            hue=cj.get("hue", 0.0),
        ),
        transforms.RandAugment(
            num_ops=int(ra.get("num_ops", 2)),
            magnitude=int(ra.get("magnitude", 9)),
        ),
        transforms.ToTensor(),
        normalize,
    ]
    # NOTE: horizontal_flip is intentionally never added (orientation-sensitive).
    return transforms.Compose(ops)


class ASLDataset(Dataset):
    """Reads (image, label) pairs from a split CSV of image_path,label,cluster_id."""

    def __init__(
        self,
        index_csv: str | Path,
        data_root: str | Path,
        class_to_idx: dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = pd.read_csv(index_csv)
        self.data_root = Path(data_root)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        with Image.open(self.data_root / row["image_path"]) as im:
            img = im.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img, self.class_to_idx[row["label"]]


def build_dataloaders(
    cfg: dict,
    index_dir: str | Path,
    data_root: str | Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """Build DataLoaders for the requested splits from cached index CSVs."""
    class_to_idx = build_class_to_idx(cfg["dataset"]["classes"])
    index_dir = Path(index_dir)
    loader_cfg = cfg["loader"]

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        train = split == "train"
        ds = ASLDataset(
            index_csv=index_dir / f"{split}.csv",
            data_root=data_root,
            class_to_idx=class_to_idx,
            transform=build_transforms(cfg, train=train),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=int(loader_cfg["batch_size"]),
            shuffle=train,
            num_workers=int(loader_cfg.get("num_workers", 0)),
            pin_memory=bool(loader_cfg.get("pin_memory", False)),
            drop_last=train,
        )
    return loaders

"""Near-duplicate detection via perceptual hashing + LSH-banded union-find.

Why (CLAUDE.md constraint #1): the dataset has near-duplicate frames. A plain
random split leaks those duplicates across train/test and inflates accuracy. We
give every image a `cluster_id` so downstream splitting can keep whole duplicate
clusters inside a single split.

Scaling: pairwise Hamming over N images is O(N^2). We avoid it by
  1. collapsing images with an *identical* phash (exact duplicates) first, then
  2. LSH banding over the remaining distinct hashes. Pigeonhole: if two hashes
     differ in <= t bits and we cut the hash into t+1 contiguous bands, at least
     one band is bit-identical, so within-threshold pairs always land in a shared
     bucket. We only compute Hamming distance for candidates inside a bucket.

Determinism: phash is deterministic; cluster ids are assigned in first-seen
order. Log hash_size / threshold / bands so a run is reproducible.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Union-Find (disjoint set) with path compression + union by rank.
# --------------------------------------------------------------------------- #
class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --------------------------------------------------------------------------- #
# Hashing.
# --------------------------------------------------------------------------- #
def _hash_one(abs_path: str, hash_size: int) -> Optional[bytes]:
    """Perceptual hash of one image -> packed bytes, or None on failure."""
    try:
        with Image.open(abs_path) as im:
            h = imagehash.phash(im, hash_size=hash_size)
    except Exception:  # corrupt/unreadable image -> caller drops it
        return None
    return np.packbits(h.hash.flatten()).tobytes()


def compute_hashes(
    abs_paths: list[str],
    hash_size: int,
    num_workers: int = 1,
) -> list[Optional[bytes]]:
    """Compute a phash (packed bytes) per path. None marks a failed image."""
    packed: list[Optional[bytes]] = [None] * len(abs_paths)
    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            it = ex.map(_hash_one, abs_paths, repeat(hash_size), chunksize=64)
            for i, res in enumerate(tqdm(it, total=len(abs_paths), desc="phash")):
                packed[i] = res
    else:
        for i, p in enumerate(tqdm(abs_paths, desc="phash")):
            packed[i] = _hash_one(p, hash_size)
    return packed


# --------------------------------------------------------------------------- #
# Clustering.
# --------------------------------------------------------------------------- #
def cluster_duplicates(
    packed: list[bytes],
    hash_size: int,
    threshold: int,
    num_bands: Optional[int] = None,
) -> np.ndarray:
    """Cluster near-duplicate images. `packed` must contain no None entries.

    Returns an int array of cluster ids aligned to `packed`.
    """
    length = hash_size * hash_size  # bits per hash
    n = len(packed)
    dsu = DSU(n)

    # Unpack bits once (bool array per image).
    bits = [np.unpackbits(np.frombuffer(p, dtype=np.uint8))[:length] for p in packed]

    # 1) Collapse exact duplicates and keep one representative per distinct hash.
    rep_of_key: dict[bytes, int] = {}
    reps: list[int] = []
    for i in range(n):
        key = packed[i]
        rep = rep_of_key.get(key)
        if rep is None:
            rep_of_key[key] = i
            reps.append(i)
        else:
            dsu.union(rep, i)

    # 2) LSH banding over distinct representatives.
    b = num_bands if num_bands else (threshold + 1)
    b = max(1, min(b, length))
    band_slices = np.array_split(np.arange(length), b)

    for band in band_slices:
        buckets: dict[bytes, list[int]] = {}
        for r in reps:
            k = bits[r][band].tobytes()
            buckets.setdefault(k, []).append(r)
        for members in buckets.values():
            m = len(members)
            if m < 2:
                continue
            for a in range(m):
                ba = bits[members[a]]
                for c in range(a + 1, m):
                    if np.count_nonzero(ba != bits[members[c]]) <= threshold:
                        dsu.union(members[a], members[c])

    # 3) Assign compact cluster ids in first-seen order (deterministic).
    root_to_id: dict[int, int] = {}
    cluster_ids = np.empty(n, dtype=np.int64)
    next_id = 0
    for i in range(n):
        root = dsu.find(i)
        cid = root_to_id.get(root)
        if cid is None:
            cid = next_id
            root_to_id[root] = cid
            next_id += 1
        cluster_ids[i] = cid
    return cluster_ids


def run_dedup(
    df: pd.DataFrame,
    data_root: str | Path,
    cfg: dict,
) -> tuple[pd.DataFrame, int]:
    """Full dedup step: hash every image, drop unreadable ones, add cluster_id.

    Args:
        df: table with an `image_path` column (relative to data_root).
        data_root: dataset root the relative paths join onto.
        cfg: the `dedup` sub-config (hash_size, hamming_threshold, num_bands,
            num_workers).

    Returns:
        (df_valid, n_dropped) where df_valid gains a `cluster_id` column and
        n_dropped is the count of images that failed to hash.
    """
    data_root = Path(data_root)
    abs_paths = [str(data_root / p) for p in df["image_path"]]

    packed = compute_hashes(
        abs_paths,
        hash_size=int(cfg["hash_size"]),
        num_workers=int(cfg.get("num_workers", 1)),
    )

    valid_mask = np.array([p is not None for p in packed], dtype=bool)
    n_dropped = int((~valid_mask).sum())

    df_valid = df.loc[valid_mask].reset_index(drop=True).copy()
    packed_valid = [p for p, ok in zip(packed, valid_mask) if ok]

    cluster_ids = cluster_duplicates(
        packed_valid,
        hash_size=int(cfg["hash_size"]),
        threshold=int(cfg["hamming_threshold"]),
        num_bands=cfg.get("num_bands"),
    )
    df_valid["cluster_id"] = cluster_ids
    return df_valid, n_dropped

"""Group-aware train/val/test split (CLAUDE.md constraint #1).

The unit of splitting is the duplicate *cluster*, never the individual image, so
no near-duplicate can straddle two splits. Within each class we greedily assign
whole clusters (largest first) to whichever split is furthest below its image
target, giving an approximate stratified 70/15/15 while keeping clusters intact.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _allocate_clusters(
    cids: list[int],
    sizes: dict[int, int],
    ratios: dict[str, float],
    rng: np.random.Generator,
    assignment: dict[int, str],
) -> None:
    """Greedy largest-first allocation of clusters to splits to hit `ratios`."""
    total = sum(sizes[c] for c in cids)
    targets = {s: r * total for s, r in ratios.items()}
    assigned = {s: 0.0 for s in ratios}

    # Deterministic tie-break, then largest cluster first.
    tiebreak = {c: rng.random() for c in cids}
    order = sorted(cids, key=lambda c: (-sizes[c], tiebreak[c]))

    for c in order:
        # Split with the largest remaining deficit (target - assigned).
        split = max(ratios, key=lambda s: targets[s] - assigned[s])
        assignment[c] = split
        assigned[split] += sizes[c]


def group_aware_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int = 0,
    stratify_by_class: bool = True,
) -> pd.DataFrame:
    """Assign each row a `split` in {train, val, test}, keeping clusters intact.

    Args:
        df: table with columns [image_path, label, cluster_id].
        ratios: image-count target fractions, e.g. {train:.7, val:.15, test:.15}.
        seed: RNG seed for deterministic tie-breaking.
        stratify_by_class: allocate per class (approx stratified) when True.

    Returns:
        A copy of df with an added `split` column.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    rng = np.random.default_rng(seed)

    # Per-cluster size and dominant label.
    grouped = df.groupby("cluster_id")
    sizes = grouped.size().to_dict()
    dominant_label = grouped["label"].agg(lambda s: s.mode().iloc[0]).to_dict()

    assignment: dict[int, str] = {}
    if stratify_by_class:
        by_class: dict[str, list[int]] = defaultdict(list)
        for cid, lbl in dominant_label.items():
            by_class[lbl].append(cid)
        for cids in by_class.values():
            _allocate_clusters(cids, sizes, ratios, rng, assignment)
    else:
        _allocate_clusters(list(sizes), sizes, ratios, rng, assignment)

    out = df.copy()
    out["split"] = out["cluster_id"].map(assignment)
    return out


def leakage_check(df: pd.DataFrame) -> None:
    """Raise if any cluster_id appears in more than one split."""
    spans = df.groupby("cluster_id")["split"].nunique()
    offenders = spans[spans > 1]
    if not offenders.empty:
        raise AssertionError(
            f"Leakage: {len(offenders)} cluster(s) span multiple splits, "
            f"e.g. {offenders.index[:5].tolist()}"
        )

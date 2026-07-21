"""Human-readable sanity report for the Phase 1 data pipeline.

Reports the numbers a skeptical reviewer will ask for: per-class counts per
split, duplicate-cluster statistics, dropped images, and an explicit leakage
verdict.
"""
from __future__ import annotations

import pandas as pd


def build_report(
    df: pd.DataFrame,
    scan_info: dict,
    n_dropped: int,
    cfg: dict,
    leakage_ok: bool,
) -> str:
    """Assemble the Phase 1 report string from the split table."""
    lines: list[str] = []
    add = lines.append

    add("=" * 70)
    add("PHASE 1 DATA REPORT - group-aware, leakage-free split")
    add("=" * 70)
    add(f"dataset:        {cfg['dataset']['name']}")
    add(f"train_root:     {scan_info['root']}")
    add(f"images scanned: {scan_info['num_images']}")
    add(f"images dropped: {n_dropped} (unreadable / failed to hash)")
    add(f"images kept:    {len(df)}")
    if scan_info.get("missing_classes"):
        add(f"WARNING missing class folders: {scan_info['missing_classes']}")
    if scan_info.get("extra_folders"):
        add(f"WARNING extra folders (ignored): {scan_info['extra_folders']}")
    add("")

    # --- Dedup stats -------------------------------------------------------
    sizes = df.groupby("cluster_id").size()
    n_clusters = int(sizes.shape[0])
    n_dup_clusters = int((sizes > 1).sum())
    n_dup_images = int(sizes[sizes > 1].sum())
    dcfg = cfg["dedup"]
    add("-" * 70)
    add("DEDUP (near-duplicate clustering)")
    add("-" * 70)
    add(f"method:            {dcfg['method']}  hash_size={dcfg['hash_size']}  "
        f"hamming_threshold={dcfg['hamming_threshold']}  "
        f"num_bands={dcfg.get('num_bands') or dcfg['hamming_threshold'] + 1}")
    add(f"total clusters:    {n_clusters}")
    add(f"duplicate clusters (size>1): {n_dup_clusters}")
    add(f"images in duplicate clusters: {n_dup_images} "
        f"({100.0 * n_dup_images / max(len(df), 1):.1f}% of kept)")
    add(f"largest cluster:   {int(sizes.max())} images")
    add("")

    # --- Split totals ------------------------------------------------------
    split_totals = df["split"].value_counts().reindex(
        ["train", "val", "test"]).fillna(0).astype(int)
    add("-" * 70)
    add("SPLIT TOTALS")
    add("-" * 70)
    total = len(df)
    for s in ["train", "val", "test"]:
        n = int(split_totals.get(s, 0))
        add(f"{s:5s}: {n:7d}  ({100.0 * n / max(total, 1):.1f}%)")
    add("")

    # --- Per-class counts per split ---------------------------------------
    pivot = (
        df.groupby(["label", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    add("-" * 70)
    add("PER-CLASS COUNTS PER SPLIT")
    add("-" * 70)
    add(pivot.to_string())
    add("")

    # --- Leakage verdict ---------------------------------------------------
    add("-" * 70)
    add(f"LEAKAGE CHECK: {'PASS - no cluster spans two splits' if leakage_ok else 'FAIL'}")
    add("-" * 70)

    return "\n".join(lines)

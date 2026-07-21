"""Phase 1 CLI: scan -> dedup -> group-aware split -> index files + report.

Usage (locally or in a Kaggle notebook cell):
    python scripts/prepare_data.py --config configs/data.yaml
    python scripts/prepare_data.py --config configs/data.yaml --force

Reads the read-only dataset from `dataset.train_root`, writes all outputs under
`output_dir` (splits, dedup cache, report, run meta). Re-running reuses the
cached dedup clusters unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make `import src...` work regardless of where the script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dedup import run_dedup  # noqa: E402
from src.data.report import build_report  # noqa: E402
from src.data.scan import scan_dataset  # noqa: E402
from src.data.split import group_aware_split, leakage_check  # noqa: E402
from src.utils.config import ensure_parent, load_config, output_path  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 leakage-free data prep.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute dedup clusters even if the cache exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["split"]["seed"])
    seed_everything(seed)

    train_root = cfg["dataset"]["train_root"]
    cache_path = output_path(cfg, cfg["dedup"]["cache"])
    index_dir = output_path(cfg, cfg["split"]["index_dir"])
    report_path = output_path(cfg, cfg["report"]["path"])

    # --- 1. Scan ----------------------------------------------------------
    print(f"[1/5] Scanning dataset at {train_root} ...")
    df, scan_info = scan_dataset(train_root, cfg["dataset"]["classes"])
    if len(df) == 0:
        raise SystemExit(f"No images found under {train_root}. Check the path.")
    print(f"      found {len(df)} images across {len(scan_info['subdirs'])} folders")
    if scan_info["missing_classes"]:
        print(f"      WARNING missing class folders: {scan_info['missing_classes']}")
    if scan_info["extra_folders"]:
        print(f"      WARNING extra folders (ignored): {scan_info['extra_folders']}")

    # --- 2. Dedup (with cache) -------------------------------------------
    n_dropped = 0
    if cache_path.exists() and not args.force:
        print(f"[2/5] Loading cached dedup clusters from {cache_path}")
        df = pd.read_csv(cache_path)
        n_dropped = int(scan_info["num_images"] - len(df))
    else:
        print("[2/5] Computing perceptual hashes + near-duplicate clusters ...")
        df, n_dropped = run_dedup(df, train_root, cfg["dedup"])
        ensure_parent(cache_path)
        df.to_csv(cache_path, index=False)
        print(f"      dropped {n_dropped} unreadable image(s); cache -> {cache_path}")

    # --- 3. Group-aware split --------------------------------------------
    print("[3/5] Building group-aware split (clusters kept intact) ...")
    df = group_aware_split(
        df,
        ratios=cfg["split"]["ratios"],
        seed=seed,
        stratify_by_class=bool(cfg["split"].get("stratify_by_class", True)),
    )

    # --- 4. Leakage check + write index files -----------------------------
    print("[4/5] Verifying no leakage and writing index files ...")
    leakage_ok = True
    try:
        leakage_check(df)
    except AssertionError as e:
        leakage_ok = False
        print(f"      {e}")

    index_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "label", "cluster_id"]
    for split in ["train", "val", "test"]:
        sub = df.loc[df["split"] == split, cols]
        sub.to_csv(index_dir / f"{split}.csv", index=False)
    print(f"      index files -> {index_dir}/(train|val|test).csv")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "seed": seed,
        "config_path": str(Path(args.config).resolve()),
        "train_root": str(train_root),
        "num_classes": cfg["dataset"]["num_classes"],
        "images_scanned": scan_info["num_images"],
        "images_dropped": n_dropped,
        "images_kept": int(len(df)),
        "num_clusters": int(df["cluster_id"].nunique()),
        "ratios": cfg["split"]["ratios"],
        "leakage_ok": leakage_ok,
    }
    with open(ensure_parent(index_dir / "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --- 5. Report --------------------------------------------------------
    print("[5/5] Writing sanity report ...\n")
    report = build_report(df, scan_info, n_dropped, cfg, leakage_ok)
    ensure_parent(report_path)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to {report_path}")

    if not leakage_ok:
        raise SystemExit("Leakage detected — see report above.")


if __name__ == "__main__":
    main()

"""Global seeding for reproducibility (CLAUDE.md: seed everything, log the seed)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy, and (if available) torch. Returns the seed.

    torch is imported lazily so this stays usable in the torch-free data
    pipeline. `deterministic` sets cudnn deterministic flags when torch is
    present.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed

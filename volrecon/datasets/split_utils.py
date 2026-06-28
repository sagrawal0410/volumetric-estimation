"""Train/val/test split helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence


def split_scenes(
    scene_ids: Sequence[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    ids = list(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def assign_split(scene_id: str, split_map: dict[str, list[str]]) -> str | None:
    for split_name, scenes in split_map.items():
        if scene_id in scenes:
            return split_name
    return None

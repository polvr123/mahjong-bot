from __future__ import annotations

import random

import numpy as np


class RandomAgent:
    """Uniformly random legal action."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, view: dict, mask: np.ndarray) -> int:
        return self.rng.choice(np.flatnonzero(mask).tolist())

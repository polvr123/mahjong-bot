from __future__ import annotations

import random

import numpy as np

from ..engine import actions as A
from ..engine.shanten import shanten_counts
from ..engine.tiles import NUM_TYPES


class HeuristicAgent:
    """Greedy shanten reduction: win when possible, riichi when tenpai, otherwise discard the
    tile that leaves the lowest shanten (ties broken randomly)."""

    name = "heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, view: dict, mask: np.ndarray) -> int:
        if mask[A.TSUMO]:
            return A.TSUMO
        if mask[A.RON]:
            return A.RON
        riichi = np.flatnonzero(mask[A.RIICHI_BASE : A.TSUMO])
        if riichi.size:
            return A.RIICHI_BASE + int(self.rng.choice(riichi.tolist()))

        counts = [0] * NUM_TYPES
        for t in view["hand"] + [view["drawn_tile"]]:
            counts[t // 4] += 1
        best, best_s = [], 99
        for t in np.flatnonzero(mask[A.DISCARD_BASE : A.RIICHI_BASE]).tolist():
            counts[t] -= 1
            s = shanten_counts(counts)
            counts[t] += 1
            if s < best_s:
                best, best_s = [t], s
            elif s == best_s:
                best.append(t)
        return A.DISCARD_BASE + self.rng.choice(best)

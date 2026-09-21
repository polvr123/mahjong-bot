"""Encode a player view (see GameState.player_view) into a flat float vector.

Layout (seat-relative: index 0 = me, 1 = the seat after me, ...):
    hand count thermometer  34x4   (count>=1..4 per tile type, includes the drawn tile)
    drawn tile one-hot      34
    pending ron tile        34
    discards per seat       4x34   (count / 4)
    dora indicator one-hot  34
    riichi flags            4
    wall remaining          1      (/ 70)
    seat wind one-hot       4
    round wind one-hot      4
    ron-phase flag          1
    furiten flag            1
    current shanten         1      ((7 - s) / 8)
    shanten after discard   34     ((7 - s) / 8 for each discardable type, else 0)
    visible copies          34     (my hand + all discards + dora indicator, / 4)
"""
from __future__ import annotations

import numpy as np

from .shanten import shanten_counts
from .tiles import EAST, NUM_TYPES

OBS_DIM = 34 * 4 + 34 + 34 + 4 * 34 + 34 + 4 + 1 + 4 + 4 + 1 + 1 + 1 + 34 + 34
_MAX_WALL = 70.0


def _onehot(tile) -> np.ndarray:
    v = np.zeros(NUM_TYPES, dtype=np.float32)
    if tile is not None:
        v[tile // 4] = 1.0
    return v


def encode_view(view: dict) -> np.ndarray:
    me = view["me"]
    hand = list(view["hand"])
    drawn = view["drawn_tile"]
    if drawn is not None:
        hand.append(drawn)
    hc = np.zeros(NUM_TYPES, dtype=np.int32)
    for t in hand:
        hc[t // 4] += 1

    parts: list[np.ndarray] = []
    therm = np.zeros((NUM_TYPES, 4), dtype=np.float32)
    for k in range(4):
        therm[:, k] = hc > k
    parts += [therm.ravel(), _onehot(drawn), _onehot(view["pending_tile"])]

    visible = hc.astype(np.float32)
    for rel in range(4):
        d = np.zeros(NUM_TYPES, dtype=np.float32)
        for t in view["discards"][(me + rel) % 4]:
            d[t // 4] += 1
        visible += d
        parts.append(d / 4.0)

    dora = np.zeros(NUM_TYPES, dtype=np.float32)
    for ind in view["dora_indicators"]:
        dora[ind // 4] = 1.0
        visible[ind // 4] += 1
    parts.append(dora)

    parts.append(np.array([view["riichi"][(me + rel) % 4] for rel in range(4)], dtype=np.float32))
    parts.append(np.array([view["wall_remaining"] / _MAX_WALL], dtype=np.float32))
    for wind in (view["seat_winds"][me], view["round_wind"]):
        w = np.zeros(4, dtype=np.float32)
        w[wind - EAST] = 1.0
        parts.append(w)
    parts.append(np.array([view["phase"] == "ron", view["furiten"]], dtype=np.float32))

    counts = hc.tolist()
    parts.append(np.array([(7 - shanten_counts(counts)) / 8.0], dtype=np.float32))
    after = np.zeros(NUM_TYPES, dtype=np.float32)
    if drawn is not None:  # only meaningful when there is a tile to discard (14 tiles)
        for t in range(NUM_TYPES):
            if counts[t]:
                counts[t] -= 1
                after[t] = (7 - shanten_counts(counts)) / 8.0
                counts[t] += 1
    parts += [after, np.clip(visible, 0, 4) / 4.0]

    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape == (OBS_DIM,)
    return obs

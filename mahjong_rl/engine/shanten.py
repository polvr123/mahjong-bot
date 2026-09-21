"""Shanten and wait helpers built on the `mahjong` library (0 = tenpai, -1 = complete hand)."""
from __future__ import annotations

from mahjong.agari import Agari
from mahjong.shanten import Shanten

from .tiles import NUM_TYPES, counts34

_shanten = Shanten()
_agari = Agari()


def shanten_counts(c34: list[int]) -> int:
    """Shanten of a 13- or 14-tile hand given as 34-type counts."""
    return _shanten.calculate_shanten(c34)


def shanten(tiles: list[int]) -> int:
    """Shanten of a 13- or 14-tile hand given as 136-encoded tiles."""
    return shanten_counts(counts34(tiles))


def is_complete_counts(c34: list[int]) -> bool:
    """True if the 14 tiles form a complete hand shape (ignores yaku)."""
    return _agari.is_agari(c34)


def waits_counts(c34_13: list[int]) -> list[int]:
    """Tile types that complete a 13-tile hand (shape only; ignores yaku and furiten)."""
    waits = []
    for t in range(NUM_TYPES):
        if c34_13[t] >= 4:
            continue
        c34_13[t] += 1
        if _agari.is_agari(c34_13):
            waits.append(t)
        c34_13[t] -= 1
    return waits

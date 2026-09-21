"""Tile encoding and the wall.

Tiles are ints 0-135 (the encoding the `mahjong` library uses): ``tile // 4`` is the
34-type index and ``tile % 4`` the copy number.

Type layout: 0-8 man, 9-17 pin, 18-26 sou, 27-30 winds (E S W N), 31-33 dragons (white, green, red).
"""
from __future__ import annotations

import random

NUM_TILES = 136
NUM_TYPES = 34
DEAD_WALL_SIZE = 14
HAND_SIZE = 13

EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30

_HONOR_NAMES = ["E", "S", "W", "N", "P", "F", "C"]  # P=white(haku), F=green(hatsu), C=red(chun)


def tile_type(tile: int) -> int:
    return tile // 4


def type_name(t: int) -> str:
    if t < 27:
        return f"{t % 9 + 1}{'mps'[t // 9]}"
    return _HONOR_NAMES[t - 27]


def tile_name(tile: int) -> str:
    return type_name(tile_type(tile))


def dora_from_indicator(indicator: int) -> int:
    """34-type of the dora, given the indicator's 34-type."""
    if indicator < 27:
        return indicator // 9 * 9 + (indicator % 9 + 1) % 9
    if indicator < 31:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


def counts34(tiles) -> list[int]:
    """How many of each of the 34 tile types appear in `tiles` (136-encoded)."""
    c = [0] * NUM_TYPES
    for t in tiles:
        c[t // 4] += 1
    return c


class Wall:
    """Shuffled 136-tile wall. The last 14 tiles are the dead wall (dora indicators)."""

    def __init__(self, rng: random.Random | None = None):
        rng = rng or random.Random()
        self.tiles: list[int] = list(range(NUM_TILES))
        rng.shuffle(self.tiles)
        self._pos = 0  # next live tile to draw

    def deal(self) -> list[list[int]]:
        """Deal 13 tiles to each of the 4 seats."""
        hands = [[] for _ in range(4)]
        for _ in range(HAND_SIZE):
            for p in range(4):
                hands[p].append(self.draw())
        return [sorted(h) for h in hands]

    @property
    def dead_wall(self) -> list[int]:
        return self.tiles[-DEAD_WALL_SIZE:]

    @property
    def dora_indicator(self) -> int:
        return self.dead_wall[4]

    @property
    def remaining(self) -> int:
        """Live-wall tiles left to draw."""
        return NUM_TILES - DEAD_WALL_SIZE - self._pos

    def is_empty(self) -> bool:
        return self.remaining <= 0

    def draw(self) -> int:
        if self.is_empty():
            raise IndexError("live wall is empty")
        t = self.tiles[self._pos]
        self._pos += 1
        return t

"""Helpers for building scripted game situations in tests."""
from __future__ import annotations

from mahjong.tile import TilesConverter

from mahjong_rl.engine.game import MahjongGame
from mahjong_rl.engine.shanten import waits_counts
from mahjong_rl.engine.state import PHASE_TURN
from mahjong_rl.engine.tiles import counts34


def ids(spec: dict, used: set[int]) -> list[int]:
    """ids(dict(man='123', pin='55'), used) -> distinct 136-ids, avoiding those in `used` (which is updated)."""
    out = []
    for t, c in enumerate(TilesConverter.string_to_34_array(**spec)):
        free = [x for x in range(4 * t, 4 * t + 4) if x not in used]
        assert len(free) >= c, "not enough copies of that tile"
        out += free[:c]
        used.update(free[:c])
    return out


def scripted_game(hand_specs: list[dict], dealer: int = 0, drawn: dict | None = None, mover: int = 0) -> MahjongGame:
    """A game where each seat holds the given 13 tiles and `mover` has just drawn `drawn` (a 1-tile spec)."""
    g = MahjongGame(dealer=dealer, seed=123)
    s = g.state
    used: set[int] = set()
    s.hands = [sorted(ids(spec, used)) for spec in hand_specs]
    for p in range(4):
        s.waits[p] = waits_counts(counts34(s.hands[p]))
    s.dora_indicators = [ids(dict(honors="4"), used)[0]]  # north indicator -> east dora
    s.discards = [[], [], [], []]
    s.riichi_declared = [False] * 4
    s.riichi_paid = [False] * 4
    s.ippatsu = [False] * 4
    s.furiten_temp = [False] * 4
    s.ron_queue, s.ron_scores, s.result = [], {}, None
    s.phase, s.current_player, s.drawn_tile = PHASE_TURN, mover, None
    if drawn is not None:
        tile = ids(drawn, used)[0]
        s.hands[mover].append(tile)
        s.drawn_tile = tile
    g._mask = None
    return g


# 13-tile hands used across tests
TENPAI_3P = dict(man="123", pin="24456", sou="78999")   # kanchan wait on 3p only; NO yaku unless riichi
JUNK_A = dict(man="147", pin="258", sou="369", honors="1234")
JUNK_B = dict(man="258", pin="147", sou="147", honors="5671")
JUNK_C = dict(man="369", pin="369", sou="258", honors="1235")

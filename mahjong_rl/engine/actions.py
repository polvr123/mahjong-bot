"""Action space and legal-action generation.

Discrete(71):
    0-33   discard a tile of that 34-type
    34-67  declare riichi, discarding a tile of that 34-type
    68     tsumo
    69     ron
    70     pass (decline a ron)
"""
from __future__ import annotations

import numpy as np

from .scoring import ScoreResult, score_hand
from .shanten import is_complete_counts, shanten_counts
from .state import PHASE_RON, PHASE_TURN, GameState
from .tiles import NUM_TYPES, counts34, type_name

DISCARD_BASE = 0
RIICHI_BASE = NUM_TYPES
TSUMO = 2 * NUM_TYPES
RON = TSUMO + 1
PASS = TSUMO + 2
NUM_ACTIONS = PASS + 1

_MIN_WALL_FOR_RIICHI = 4


def action_name(a: int) -> str:
    if a < RIICHI_BASE:
        return f"discard {type_name(a)}"
    if a < TSUMO:
        return f"riichi+discard {type_name(a - RIICHI_BASE)}"
    return {TSUMO: "tsumo", RON: "ron", PASS: "pass"}[a]


def evaluate_win(s: GameState, p: int, tile: int, tsumo: bool) -> ScoreResult:
    """Score player p winning on `tile`. For tsumo the hand already contains `tile` (14 tiles)."""
    tiles = list(s.hands[p]) if tsumo else s.hands[p] + [tile]
    no_discards_yet = not any(s.discards)
    return score_hand(
        tiles,
        tile,
        is_tsumo=tsumo,
        is_riichi=s.riichi_declared[p],
        is_ippatsu=s.ippatsu[p],
        dora_indicators=s.dora_indicators,
        player_wind=s.seat_winds[p],
        round_wind=s.round_wind,
        is_haitei=tsumo and s.wall.is_empty(),
        is_houtei=(not tsumo) and s.wall.is_empty(),
        is_tenhou=tsumo and p == s.dealer and no_discards_yet,
        is_chiihou=tsumo and p != s.dealer and not s.discards[p],
    )


def can_tsumo(s: GameState, p: int) -> ScoreResult | None:
    """Score of a tsumo win for p on their drawn tile, or None if they can't win."""
    if not is_complete_counts(counts34(s.hands[p])):  # cheap shape check before the full scorer
        return None
    r = evaluate_win(s, p, s.drawn_tile, tsumo=True)
    return r if r.is_valid_win else None


def riichi_discard_types(s: GameState, p: int) -> list[int]:
    """34-types that can be discarded to declare riichi (the resulting 13 tiles are tenpai)."""
    if s.riichi_declared[p] or s.wall.remaining < _MIN_WALL_FOR_RIICHI:
        return []
    c = counts34(s.hands[p])
    if shanten_counts(c) > 0:  # no discard can reach tenpai
        return []
    out = []
    for t in range(NUM_TYPES):
        if c[t]:
            c[t] -= 1
            if shanten_counts(c) == 0:
                out.append(t)
            c[t] += 1
    return out


def legal_action_mask(s: GameState, p: int | None = None) -> np.ndarray:
    """Boolean mask over the action space for the player who has to act."""
    p = s.decision_player if p is None else p
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if s.done or p != s.decision_player:
        return mask
    if s.phase == PHASE_RON:
        mask[RON] = mask[PASS] = True
        return mask
    assert s.phase == PHASE_TURN
    if can_tsumo(s, p) is not None:
        mask[TSUMO] = True
    if s.riichi_declared[p]:
        mask[DISCARD_BASE + s.drawn_tile // 4] = True  # hand is locked: only the drawn tile can go
        return mask
    for t in {t // 4 for t in s.hands[p]}:
        mask[DISCARD_BASE + t] = True
    for t in riichi_discard_types(s, p):
        mask[RIICHI_BASE + t] = True
    return mask


def legal_actions(s: GameState, p: int | None = None) -> list[int]:
    return [int(a) for a in np.flatnonzero(legal_action_mask(s, p))]

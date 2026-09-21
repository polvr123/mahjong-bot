"""Thin wrapper over `mahjong`'s HandCalculator. No yaku logic lives here."""
from __future__ import annotations

from dataclasses import dataclass, field

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig

from .tiles import EAST

_calc = HandCalculator()


@dataclass
class ScoreResult:
    is_valid_win: bool
    han: int = 0
    fu: int = 0
    points: int = 0  # total the winner receives from the hand (excludes riichi sticks)
    yaku: list[str] = field(default_factory=list)
    cost: dict = field(default_factory=dict)  # raw payment breakdown from the library
    error: str | None = None

    def as_tuple(self):
        return self.is_valid_win, self.han, self.fu, self.points, self.yaku


def score_hand(
    tiles: list[int],
    win_tile: int,
    *,
    is_tsumo: bool,
    is_riichi: bool = False,
    is_ippatsu: bool = False,
    dora_indicators: list[int] | None = None,
    player_wind: int,
    round_wind: int,
    is_haitei: bool = False,
    is_houtei: bool = False,
    is_tenhou: bool = False,
    is_chiihou: bool = False,
) -> ScoreResult:
    """Score a closed 14-tile hand (13 + `win_tile`, 136-encoded).

    `player_wind` / `round_wind` are wind type indices (27-30). A hand that has a complete
    shape but no yaku comes back with is_valid_win=False (error "no_yaku").
    """
    config = HandConfig(
        is_tsumo=is_tsumo,
        is_riichi=is_riichi,
        is_ippatsu=is_ippatsu,
        is_haitei=is_haitei,
        is_houtei=is_houtei,
        is_tenhou=is_tenhou,
        is_chiihou=is_chiihou,
        player_wind=player_wind,
        round_wind=round_wind,
    )
    r = _calc.estimate_hand_value(tiles, win_tile, dora_indicators=dora_indicators or [], config=config)
    if r.error:
        return ScoreResult(False, error=r.error)
    cost = r.cost or {}
    points = _tsumo_total(cost, player_wind) if is_tsumo else int(cost.get("total", 0))
    return ScoreResult(True, r.han, r.fu, points, [str(y) for y in r.yaku], cost)


def _tsumo_total(cost: dict, player_wind: int) -> int:
    main, add = cost["main"], cost["additional"]
    # dealer tsumo: three payers each pay `main`; non-dealer: dealer pays main, two others pay additional
    return main * 3 if player_wind == EAST else main + add * 2


def payments(result: ScoreResult, winner: int, dealer: int, loser: int | None) -> list[int]:
    """Per-seat point changes for a win (no honba/riichi sticks). `loser` is None for tsumo."""
    d = [0, 0, 0, 0]
    if loser is not None:
        d[winner] += result.points
        d[loser] -= result.points
        return d
    main, add = result.cost["main"], result.cost["additional"]
    for p in range(4):
        if p == winner:
            continue
        pay = main if (winner == dealer or p == dealer) else add
        d[p] -= pay
        d[winner] += pay
    return d

"""Turn loop for a single no-calls hand: draw -> (tsumo | discard/riichi) -> ron window -> next seat."""
from __future__ import annotations

import random

import numpy as np

from . import actions as A
from .scoring import payments
from .shanten import waits_counts
from .state import PHASE_DONE, PHASE_RON, PHASE_TURN, GameState, HandResult
from .tiles import EAST, Wall, counts34

RIICHI_STICK = 1000


class IllegalAction(ValueError):
    pass


class MahjongGame:
    """One hand of simplified riichi mahjong (no calls, single dora, no ura-dora).

    Usage::

        g = MahjongGame(dealer=0, seed=1)
        while not g.done:
            p = g.decision_player
            g.step(p, choose(g.legal_actions()))
    """

    def __init__(self, dealer: int = 0, seed: int | None = None, rng: random.Random | None = None):
        self.rng = rng or random.Random(seed)
        wall = Wall(self.rng)
        seat_winds = [EAST + (p - dealer) % 4 for p in range(4)]
        self.state = GameState(
            wall=wall,
            dealer=dealer,
            hands=wall.deal(),
            dora_indicators=[wall.dora_indicator],
            seat_winds=seat_winds,
        )
        for p in range(4):
            self.state.waits[p] = waits_counts(counts34(self.state.hands[p]))
        self._mask: np.ndarray | None = None  # cached legal mask for the current decision
        self._draw(dealer)

    # ------------------------------------------------------------------ queries
    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def result(self) -> HandResult | None:
        return self.state.result

    @property
    def decision_player(self) -> int:
        return self.state.decision_player

    def legal_mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = A.legal_action_mask(self.state)
        return self._mask

    def legal_actions(self) -> list[int]:
        return [int(a) for a in np.flatnonzero(self.legal_mask())]

    # ------------------------------------------------------------------ transitions
    def step(self, player: int, action: int) -> None:
        s = self.state
        if s.done:
            raise IllegalAction("hand is over")
        if player != s.decision_player:
            raise IllegalAction(f"seat {player} is not to act (seat {s.decision_player} is)")
        if not self.legal_mask()[action]:
            raise IllegalAction(f"action '{A.action_name(action)}' is not legal for seat {player}")
        self._mask = None

        if s.phase == PHASE_RON:
            self._step_ron(player, action)
        elif action == A.TSUMO:
            self._finish_win(player, loser=None, score=A.can_tsumo(s, player))
        else:
            self._step_discard(player, action)

    def _draw(self, p: int) -> None:
        s = self.state
        tile = s.wall.draw()
        s.hands[p].append(tile)
        s.drawn_tile = tile
        s.current_player = p
        s.phase = PHASE_TURN
        s.last_discard = s.last_discarder = None
        s.ron_queue = []
        s.ron_scores = {}

    def _step_discard(self, p: int, action: int) -> None:
        s = self.state
        riichi = action >= A.RIICHI_BASE
        t = action - A.RIICHI_BASE if riichi else action - A.DISCARD_BASE
        # physical tile: prefer the drawn tile (tsumogiri), else the first copy in hand
        tile = s.drawn_tile if s.drawn_tile // 4 == t else next(x for x in s.hands[p] if x // 4 == t)

        was_riichi = s.riichi_declared[p]
        s.hands[p].remove(tile)
        s.hands[p].sort()
        s.discards[p].append(tile)
        s.drawn_tile = None
        if was_riichi:
            s.ippatsu[p] = False  # ippatsu window closes at the player's next discard
        else:
            s.furiten_temp[p] = False  # temporary furiten ends at your own discard
        if riichi:
            s.riichi_declared[p] = True
            s.ippatsu[p] = True
        s.waits[p] = waits_counts(counts34(s.hands[p]))
        s.last_discard, s.last_discarder = tile, p

        # ron offers in turn order from the discarder (head-bump: first taker wins)
        s.ron_queue, s.ron_scores = [], {}
        for q in ((p + i) % 4 for i in range(1, 4)):
            if tile // 4 in s.waits[q] and not s.is_furiten(q):
                score = A.evaluate_win(s, q, tile, tsumo=False)
                if score.is_valid_win:
                    s.ron_queue.append(q)
                    s.ron_scores[q] = score
        if s.ron_queue:
            s.phase = PHASE_RON
        else:
            self._advance(p)

    def _step_ron(self, q: int, action: int) -> None:
        s = self.state
        if action == A.RON:
            self._finish_win(q, loser=s.last_discarder, score=s.ron_scores[q])
            return
        s.furiten_temp[q] = True  # passed on a winning tile
        s.ron_queue.pop(0)
        if not s.ron_queue:
            self._advance(s.last_discarder)

    def _advance(self, discarder: int) -> None:
        """The discard went unclaimed: settle a riichi stick, then next draw or exhaustive draw."""
        s = self.state
        if s.riichi_declared[discarder]:
            s.riichi_paid[discarder] = True
        if s.wall.is_empty():
            s.result = HandResult(kind="draw")  # v1: no noten payments, sticks returned
            s.phase = PHASE_DONE
            return
        self._draw((discarder + 1) % 4)

    def _finish_win(self, winner: int, loser: int | None, score) -> None:
        s = self.state
        deltas = payments(score, winner, s.dealer, loser)
        for p in range(4):
            if s.riichi_paid[p]:
                deltas[p] -= RIICHI_STICK
                deltas[winner] += RIICHI_STICK
        s.result = HandResult(
            kind="ron" if loser is not None else "tsumo",
            winner=winner,
            loser=loser,
            han=score.han,
            fu=score.fu,
            points=score.points,
            yaku=score.yaku,
            deltas=deltas,
        )
        s.phase = PHASE_DONE

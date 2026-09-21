"""GameState: everything that defines one hand in progress."""
from __future__ import annotations

from dataclasses import dataclass, field

from .tiles import EAST, Wall

PHASE_TURN = "turn"  # current_player just drew: may tsumo / discard / riichi
PHASE_RON = "ron"    # someone discarded: ron_queue[0] may ron or pass
PHASE_DONE = "done"


@dataclass
class HandResult:
    kind: str                       # "tsumo" | "ron" | "draw"
    winner: int | None = None
    loser: int | None = None        # the discarder, for ron
    han: int = 0
    fu: int = 0
    points: int = 0
    yaku: list[str] = field(default_factory=list)
    deltas: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # per-seat point change, zero-sum


@dataclass
class GameState:
    wall: Wall
    dealer: int
    hands: list[list[int]]                          # 13 tiles each; 14 for the player who just drew
    discards: list[list[int]] = field(default_factory=lambda: [[], [], [], []])
    melds: list[list] = field(default_factory=lambda: [[], [], [], []])  # unused in v1 (no calls)
    dora_indicators: list[int] = field(default_factory=list)
    riichi_declared: list[bool] = field(default_factory=lambda: [False] * 4)
    riichi_paid: list[bool] = field(default_factory=lambda: [False] * 4)  # stick placed (discard not ronned)
    ippatsu: list[bool] = field(default_factory=lambda: [False] * 4)
    furiten_temp: list[bool] = field(default_factory=lambda: [False] * 4)
    waits: list[list[int]] = field(default_factory=lambda: [[], [], [], []])  # types completing each 13-tile hand
    current_player: int = 0
    drawn_tile: int | None = None
    phase: str = PHASE_TURN
    last_discard: int | None = None
    last_discarder: int | None = None
    ron_queue: list[int] = field(default_factory=list)
    ron_scores: dict = field(default_factory=dict)  # player -> ScoreResult for pending ron offers
    round_wind: int = EAST
    seat_winds: list[int] = field(default_factory=lambda: [EAST, EAST + 1, EAST + 2, EAST + 3])
    result: HandResult | None = None

    @property
    def done(self) -> bool:
        return self.phase == PHASE_DONE

    @property
    def decision_player(self) -> int:
        """Whose action is needed right now."""
        return self.ron_queue[0] if self.phase == PHASE_RON else self.current_player

    def is_furiten(self, p: int) -> bool:
        """Furiten: can't ron if any of your waits is in your own discards, or you passed a ron since."""
        if self.furiten_temp[p]:
            return True
        types = {t // 4 for t in self.discards[p]}
        return any(w in types for w in self.waits[p])

    def player_view(self, p: int) -> dict:
        """JSON-serialisable view of the state as seen by seat `p` (no hidden information)."""
        drawn = self.drawn_tile if (self.phase == PHASE_TURN and self.current_player == p) else None
        return {
            "me": p,
            "dealer": self.dealer,
            "round_wind": self.round_wind,
            "seat_winds": list(self.seat_winds),
            "hand": sorted(h for h in self.hands[p] if h != drawn),  # excludes the just-drawn tile
            "drawn_tile": drawn,
            "discards": [list(d) for d in self.discards],
            "dora_indicators": list(self.dora_indicators),
            "riichi": list(self.riichi_declared),
            "wall_remaining": self.wall.remaining,
            "phase": self.phase,
            "pending_tile": self.last_discard if self.phase == PHASE_RON else None,
            "pending_from": self.last_discarder if self.phase == PHASE_RON else None,
            "furiten": self.is_furiten(p),
        }

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Anything that picks an action from a player view (GameState.player_view) and a legal-action mask."""

    def act(self, view: dict, mask: np.ndarray) -> int: ...

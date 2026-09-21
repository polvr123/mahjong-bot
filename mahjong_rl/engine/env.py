"""Gymnasium wrapper: one learner seat, three opponents driven by Agents.

The env only surfaces the learner's decision points (its own turns and ron/pass offers);
opponent seats are advanced internally. Use with sb3-contrib's MaskablePPO: the env exposes
`action_masks()`.

Reward = (points change of the hand / reward_scale, clipped)
       + shaping_coef * (shanten reduction) after each of the learner's discards.
`info` carries `raw_reward` and `shaped_reward` separately on every step.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import actions as A
from .game import MahjongGame
from .observation import OBS_DIM, encode_view
from .shanten import shanten
from .state import PHASE_TURN


class MahjongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponents=None, shaping_coef: float = 0.01, reward_scale: float = 8000.0, reward_clip: float = 4.0):
        """`opponents`: an Agent used for the 3 other seats, or a zero-arg callable returning an Agent
        (called once per opponent seat at each reset, e.g. to sample from a pool of checkpoints)."""
        super().__init__()
        from ..agents.random_agent import RandomAgent

        self._opponent_source = opponents if opponents is not None else RandomAgent()
        self.shaping_coef = shaping_coef
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(A.NUM_ACTIONS)
        self.game: MahjongGame | None = None
        self.agent_seat = 0
        self._opponents: list = [None] * 4
        self._prev_shanten = 0
        self._ep_raw = 0.0
        self._ep_shaped = 0.0

    # ------------------------------------------------------------------ helpers
    def set_opponents(self, opponents) -> None:
        """Swap the opponent source; takes effect from the next reset()."""
        self._opponent_source = opponents

    def _make_opponent(self):
        src = self._opponent_source
        return src() if callable(src) else src  # Agent instances aren't callable; factories (incl. classes) are

    def _view(self) -> dict:
        return self.game.state.player_view(self.agent_seat)

    def _obs(self) -> np.ndarray:
        return encode_view(self._view())

    def action_masks(self) -> np.ndarray:
        return self.game.legal_mask() if not self.game.done else np.ones(A.NUM_ACTIONS, dtype=bool)

    def _play_opponents(self) -> None:
        g = self.game
        while not g.done and g.decision_player != self.agent_seat:
            p = g.decision_player
            g.step(p, self._opponents[p].act(g.state.player_view(p), g.legal_mask()))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        """options: {"seat": learner seat, "dealer": dealer seat}; both random by default."""
        super().reset(seed=seed)
        options = options or {}
        while True:  # retry in the rare case an opponent wins before the learner ever acts
            self.agent_seat = int(options.get("seat", self.np_random.integers(4)))
            dealer = int(options.get("dealer", self.np_random.integers(4)))
            self.game = MahjongGame(dealer=dealer, seed=int(self.np_random.integers(2**31 - 1)))
            self._opponents = [self._make_opponent() for _ in range(4)]
            self._play_opponents()
            if not self.game.done:
                break
            options = {k: v for k, v in options.items() if k != "dealer"}
        self._prev_shanten = shanten(self._view()["hand"])  # 13 tiles (drawn tile excluded)
        self._ep_raw = self._ep_shaped = 0.0
        return self._obs(), {"agent_seat": self.agent_seat, "dealer": self.game.state.dealer}

    def step(self, action: int):
        g = self.game
        me = self.agent_seat
        was_turn = g.state.phase == PHASE_TURN
        g.step(me, int(action))  # raises IllegalAction if the action was masked out

        shaped = 0.0
        if was_turn and action != A.TSUMO:  # the learner discarded: its hand is back to 13 tiles
            new = shanten(g.state.hands[me])
            shaped = self.shaping_coef * (self._prev_shanten - new)
            self._prev_shanten = new
        self._play_opponents()

        raw = 0.0
        info: dict = {}
        if g.done:
            res = g.result
            raw = float(np.clip(res.deltas[me] / self.reward_scale, -self.reward_clip, self.reward_clip))
            info.update(
                outcome="win" if res.winner == me else ("draw" if res.kind == "draw" else "loss"),
                won=res.winner == me,
                dealt_in=res.kind == "ron" and res.loser == me,
                result_kind=res.kind,
                points=res.deltas[me],
                agent_seat=me,
            )
        self._ep_raw += raw
        self._ep_shaped += shaped
        info.update(raw_reward=raw, shaped_reward=shaped)
        if g.done:
            info.update(episode_raw_return=self._ep_raw, episode_shaped_return=self._ep_shaped)
        return self._obs(), raw + shaped, g.done, False, info

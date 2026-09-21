"""Self-play opponents: a pool of frozen policy snapshots mixed with the heuristic bot.

The learner (one seat per episode) plays against three opponents that are each sampled, at
every reset, from:  the latest snapshot  |  an older snapshot  |  the heuristic bot.
Mixing in older snapshots and the heuristic keeps the policy from overfitting to only itself.
"""
from __future__ import annotations

import random
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from ..agents.heuristic_agent import HeuristicAgent
from ..agents.ppo_agent import PPOAgent


class OpponentPool:
    def __init__(self, p_latest: float = 0.5, p_older: float = 0.25, p_heuristic: float = 0.25, max_older: int = 8, seed: int = 0):
        self.weights = {"latest": p_latest, "older": p_older, "heuristic": p_heuristic}
        self.latest: PPOAgent | None = None
        self.older: deque[PPOAgent] = deque(maxlen=max_older)
        self.rng = random.Random(seed)

    def add_snapshot(self, policy) -> None:
        if self.latest is not None:
            self.older.append(self.latest)
        self.latest = PPOAgent.snapshot(policy, deterministic=False)

    def sample(self):
        """Factory handed to MahjongEnv(opponents=...): called once per opponent seat per reset."""
        kinds, weights = ["heuristic"], [self.weights["heuristic"]]
        if self.latest is not None:
            kinds.append("latest"); weights.append(self.weights["latest"])
        if self.older:
            kinds.append("older"); weights.append(self.weights["older"])
        kind = self.rng.choices(kinds, weights)[0]
        if kind == "latest":
            return self.latest
        if kind == "older":
            return self.rng.choice(list(self.older))
        return HeuristicAgent(self.rng.randrange(2**31))


class SelfPlayCallback(BaseCallback):
    """Refreshes the pool's snapshots every `every` PPO iterations (one iteration = one rollout + update)."""

    def __init__(self, pool: OpponentPool, every: int = 5):
        super().__init__()
        self.pool, self.every, self.iteration = pool, every, 0

    def _on_training_start(self) -> None:
        self.pool.add_snapshot(self.model.policy)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        if self.iteration % self.every == 0:
            self.pool.add_snapshot(self.model.policy)

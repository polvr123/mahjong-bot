"""Training callbacks: log raw vs shaped reward, and run the periodic win-rate evaluation."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..agents.heuristic_agent import HeuristicAgent
from ..agents.ppo_agent import PPOAgent
from .evaluate import append_csv, evaluate


class RewardLogCallback(BaseCallback):
    """Logs, per rollout, the mean raw (hand outcome) and shaped (shanten) episode returns and the
    learner's win / deal-in rate against the *training* opponent mix."""

    def __init__(self):
        super().__init__()
        self._raw, self._shaped, self._won, self._dealt_in, self._draw = [], [], [], [], []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_raw_return" in info:
                self._raw.append(info["episode_raw_return"])
                self._shaped.append(info["episode_shaped_return"])
                self._won.append(info["won"])
                self._dealt_in.append(info["dealt_in"])
                self._draw.append(info["outcome"] == "draw")
        return True

    def _on_rollout_end(self) -> None:
        if not self._raw:
            return
        rec = self.logger.record
        rec("hands/raw_return_mean", float(np.mean(self._raw)))
        rec("hands/shaped_return_mean", float(np.mean(self._shaped)))
        rec("hands/win_rate", float(np.mean(self._won)))
        rec("hands/deal_in_rate", float(np.mean(self._dealt_in)))
        rec("hands/draw_rate", float(np.mean(self._draw)))
        rec("hands/episodes", len(self._raw))
        self._raw, self._shaped, self._won, self._dealt_in, self._draw = [], [], [], [], []


class EvalCallback(BaseCallback):
    """Every `every` iterations (and once before training): play the current policy (greedy) against
    3 heuristic bots for `episodes` hands, append a row to <run_dir>/eval.csv, and save a checkpoint."""

    def __init__(self, run_dir: str, every: int = 10, episodes: int = 400, eval_seed: int = 0):
        super().__init__()
        self.run_dir, self.every, self.episodes, self.eval_seed = run_dir, every, episodes, eval_seed
        self.iteration = 0
        self.best = -1.0
        self.t0 = time.time()
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        if self.iteration % self.every == 0:
            self._evaluate()

    def _on_training_end(self) -> None:
        if self.iteration % self.every != 0:  # make sure the final policy is evaluated too
            self._evaluate()

    def _evaluate(self) -> None:
        agent = PPOAgent.snapshot(self.model.policy, deterministic=True)
        res = evaluate(agent, HeuristicAgent, episodes=self.episodes, seed=self.eval_seed)
        row = {"iteration": self.iteration, "timesteps": self.num_timesteps, "elapsed_s": round(time.time() - self.t0), **res}
        append_csv(os.path.join(self.run_dir, "eval.csv"), row)
        for k in ("win_rate", "draw_rate", "deal_in_rate", "avg_points"):
            self.logger.record(f"eval/{k}", res[k])
        ckpt = os.path.join(self.run_dir, "checkpoints", f"iter_{self.iteration:05d}")
        self.model.save(ckpt)
        self.model.save(os.path.join(self.run_dir, "latest"))
        if res["win_rate"] > self.best:
            self.best = res["win_rate"]
            self.model.save(os.path.join(self.run_dir, "best"))
        print(f"[eval] iter {self.iteration:>4}  steps {self.num_timesteps:>9}  win {res['win_rate']:.3f}  "
              f"deal-in {res['deal_in_rate']:.3f}  draw {res['draw_rate']:.3f}  pts {res['avg_points']:+.0f}", flush=True)

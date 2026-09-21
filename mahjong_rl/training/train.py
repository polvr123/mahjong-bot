"""PPO self-play training entrypoint.

    python -m mahjong_rl.training.train --run-name v1 --timesteps 2000000

Outputs to runs/<run-name>/: eval.csv (win-rate curve), checkpoints/, latest.zip, best.zip,
baseline.json (heuristic-vs-heuristic reference), config.json, progress.csv + TensorBoard events.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from ..agents.heuristic_agent import HeuristicAgent
from ..engine.env import MahjongEnv
from .callbacks import EvalCallback, RewardLogCallback
from .evaluate import evaluate
from .self_play import OpponentPool, SelfPlayCallback


def make_env_fn(pool: OpponentPool, shaping_coef: float):
    def _make():
        env = MahjongEnv(opponents=pool.sample, shaping_coef=shaping_coef)
        return ActionMasker(env, lambda e: e.unwrapped.action_masks())

    return _make


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Self-play PPO training for the riichi mahjong env")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=512, help="steps per env per iteration")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--net", type=int, nargs="+", default=[256, 256], help="hidden layer sizes")
    ap.add_argument("--shaping-coef", type=float, default=0.01, help="reward per shanten improvement (0 = off)")
    ap.add_argument("--snapshot-every", type=int, default=5, help="iterations between opponent-pool snapshots")
    ap.add_argument("--p-latest", type=float, default=0.5)
    ap.add_argument("--p-older", type=float, default=0.25)
    ap.add_argument("--p-heuristic", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=10, help="iterations between evaluations")
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(1)  # tiny nets: multi-threading only adds overhead
    run_dir = os.path.join(a.runs_dir, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    # reference line for the plot: what does a heuristic bot win against 3 heuristic bots?
    base = evaluate(HeuristicAgent(a.seed), HeuristicAgent, episodes=a.eval_episodes, seed=a.seed + 1)
    with open(os.path.join(run_dir, "baseline.json"), "w") as f:
        json.dump(base, f, indent=2)
    print(f"[baseline] heuristic vs 3 heuristics: win {base['win_rate']:.3f} draw {base['draw_rate']:.3f}", flush=True)

    pool = OpponentPool(a.p_latest, a.p_older, a.p_heuristic, seed=a.seed)
    venv = make_vec_env(make_env_fn(pool, a.shaping_coef), n_envs=a.n_envs, seed=a.seed, vec_env_cls=DummyVecEnv)
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=a.lr,
        n_steps=a.n_steps,
        batch_size=a.batch_size,
        n_epochs=a.n_epochs,
        gamma=a.gamma,
        ent_coef=a.ent_coef,
        policy_kwargs=dict(net_arch=dict(pi=a.net, vf=a.net)),
        seed=a.seed,
        device="cpu",
        verbose=0,
    )
    model.set_logger(configure(run_dir, ["csv", "tensorboard"]))  # progress.csv + TensorBoard events
    t = time.time()
    model.learn(
        a.timesteps,
        callback=[
            SelfPlayCallback(pool, a.snapshot_every),
            RewardLogCallback(),
            EvalCallback(run_dir, a.eval_every, a.eval_episodes),
        ],
    )
    model.save(os.path.join(run_dir, "final"))
    print(f"[done] {a.timesteps} steps in {time.time() - t:.0f}s -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

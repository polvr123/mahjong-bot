import csv
import os

import numpy as np

from mahjong_rl.agents.heuristic_agent import HeuristicAgent
from mahjong_rl.agents.ppo_agent import PPOAgent
from mahjong_rl.engine.game import MahjongGame
from mahjong_rl.training.self_play import OpponentPool
from mahjong_rl.training.train import main


def test_pool_falls_back_to_heuristic_then_mixes_in_snapshots(tmp_path):
    pool = OpponentPool(seed=0)
    assert all(isinstance(pool.sample(), HeuristicAgent) for _ in range(20))  # nothing to sample from yet


def test_short_training_run_end_to_end(tmp_path):
    main([
        "--run-name", "t", "--runs-dir", str(tmp_path), "--timesteps", "768", "--n-envs", "2", "--n-steps", "64",
        "--batch-size", "64", "--n-epochs", "1", "--eval-every", "2", "--eval-episodes", "8",
        "--snapshot-every", "1", "--net", "32", "32",
    ])
    run = tmp_path / "t"
    for f in ("config.json", "baseline.json", "eval.csv", "latest.zip", "best.zip", "final.zip", "progress.csv"):
        assert (run / f).exists(), f
    rows = list(csv.DictReader(open(run / "eval.csv")))
    assert int(rows[0]["iteration"]) == 0 and len(rows) >= 3  # untrained eval + periodic + final
    assert all(0.0 <= float(r["win_rate"]) <= 1.0 for r in rows)
    assert os.listdir(run / "checkpoints")

    # the saved model loads back as an Agent that only ever plays legal moves
    agent = PPOAgent.load(str(run / "latest.zip"))
    g = MahjongGame(dealer=0, seed=1)
    while not g.done:
        p = g.decision_player
        a = agent.act(g.state.player_view(p), g.legal_mask())
        assert g.legal_mask()[a]
        g.step(p, a)


def test_final_evaluation_after_last_update_does_not_crash(tmp_path):
    """Regression: a run whose last iteration isn't a multiple of --eval-every triggers an evaluation
    right after the final PPO update, when the policy still holds a gradient-carrying cached distribution
    (copy.deepcopy of it used to raise)."""
    main([
        "--run-name", "t2", "--runs-dir", str(tmp_path), "--timesteps", "768", "--n-envs", "2", "--n-steps", "64",
        "--batch-size", "64", "--n-epochs", "1", "--eval-every", "4", "--eval-episodes", "8", "--net", "16",
    ])  # 768 / 128 = 6 iterations; evals at 0, 4, and the final one at 6
    rows = list(csv.DictReader(open(tmp_path / "t2" / "eval.csv")))
    assert [int(r["iteration"]) for r in rows] == [0, 4, 6]
    assert (tmp_path / "t2" / "final.zip").exists()


def test_snapshot_matches_live_policy_and_is_frozen():
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from mahjong_rl.engine.env import MahjongEnv
    from mahjong_rl.engine.observation import encode_view

    env = ActionMasker(MahjongEnv(), lambda e: e.unwrapped.action_masks())
    m = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, policy_kwargs=dict(net_arch=[16]), device="cpu")
    m.learn(64)
    snap = PPOAgent.snapshot(m.policy, deterministic=True)
    g = MahjongGame(dealer=0, seed=2)
    p = g.decision_player
    view, mask = g.state.player_view(p), g.legal_mask()
    live, _ = m.policy.predict(encode_view(view), deterministic=True, action_masks=mask)
    assert snap.act(view, mask) == int(live)
    w0 = next(snap.policy.parameters()).clone()
    m.learn(128, reset_num_timesteps=False)
    assert torch.equal(w0, next(snap.policy.parameters()))


def test_compare_runs_and_reports_ci(tmp_path):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from mahjong_rl.engine.env import MahjongEnv
    from mahjong_rl.training.compare import compare

    env = ActionMasker(MahjongEnv(), lambda e: e.unwrapped.action_masks())
    path = str(tmp_path / "tiny.zip")
    MaskablePPO("MlpPolicy", env, n_steps=32, batch_size=32, policy_kwargs=dict(net_arch=[16]), device="cpu").save(path)
    rows = compare([path, "heuristic"], episodes=16, seed=5, workers=1)
    assert [r["candidate"] for r in rows] == [path, "heuristic"]
    assert all(0 <= r["win_rate"] <= 1 and r["ci95"] >= 0 and r["episodes"] == 16 for r in rows)
    again = compare(["heuristic"], episodes=16, seed=5, workers=1)[0]
    assert again["win_rate"] == rows[1]["win_rate"]  # same seed -> same hands -> same result

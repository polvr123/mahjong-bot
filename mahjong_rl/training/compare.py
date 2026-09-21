"""Compare checkpoints on a large, fixed set of hands against 3 heuristic bots (candidates run in parallel).

    python -m mahjong_rl.training.compare runs/shaped5/best.zip runs/shaped5/latest.zip heuristic \
        --episodes 2000 --seed 111 --out runs/compare_seed111.csv

Every candidate plays the *same* hands (same seed), so the comparison is paired. `heuristic` is the
heuristic bot itself in the agent seat (the "equal skill" reference). Error bars are 95% binomial intervals.
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

from ..agents.heuristic_agent import HeuristicAgent
from .evaluate import append_csv, evaluate


def _load(spec: str):
    if spec == "heuristic":
        return HeuristicAgent(0)
    from ..agents.ppo_agent import PPOAgent

    return PPOAgent.load(spec, deterministic=True)


def run_one(args: tuple) -> dict:
    spec, episodes, seed = args
    import torch

    torch.set_num_threads(1)
    res = evaluate(_load(spec), HeuristicAgent, episodes=episodes, seed=seed)
    res["ci95"] = 1.96 * math.sqrt(res["win_rate"] * (1 - res["win_rate"]) / episodes)
    return {"candidate": spec, "seed": seed, **res}


def compare(specs: list[str], episodes: int, seed: int, workers: int = 8) -> list[dict]:
    jobs = [(s, episodes, seed) for s in specs]
    if workers <= 1:
        return [run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(run_one, jobs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+", help="checkpoint .zip paths, or 'heuristic'")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=111, help="hand seed; use a value the training evals never used (they used 0)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="append results to this CSV")
    a = ap.parse_args(argv)
    rows = compare(a.candidates, a.episodes, a.seed, a.workers)
    rows.sort(key=lambda r: -r["win_rate"])
    print(f"{'candidate':<46} {'win rate':>16} {'deal-in':>8} {'draw':>6} {'avg pts':>8}   ({a.episodes} hands, seed {a.seed})")
    for r in rows:
        print(f"{os.path.relpath(r['candidate']) if r['candidate'] != 'heuristic' else 'heuristic (reference)':<46} "
              f"{r['win_rate']:>8.3f} ±{r['ci95']:.3f} {r['deal_in_rate']:>8.3f} {r['draw_rate']:>6.3f} {r['avg_points']:>+8.0f}")
        if a.out:
            append_csv(a.out, r)


if __name__ == "__main__":
    main()

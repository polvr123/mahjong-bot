"""Tournament runner: one agent under test vs three copies of a baseline, seat rotated every episode."""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable

from ..agents.heuristic_agent import HeuristicAgent
from ..agents.random_agent import RandomAgent
from ..engine.game import MahjongGame


def play_hand(agents: list, dealer: int, seed: int):
    """Play one hand with one Agent per seat; returns the HandResult."""
    g = MahjongGame(dealer=dealer, seed=seed)
    while not g.done:
        p = g.decision_player
        g.step(p, agents[p].act(g.state.player_view(p), g.legal_mask()))
    return g.result


def evaluate(agent, baseline_factory: Callable[..., object] = HeuristicAgent, episodes: int = 200, seed: int = 0) -> dict:
    """`agent` sits at seat (i % 4) against 3 baselines; the dealer also rotates, so every
    seat/dealer combination is covered evenly. `baseline_factory(seed=...)` builds each baseline bot;
    with the same `seed` the hands and the baselines' tie-breaks repeat exactly."""
    wins = draws = dealt_in = 0
    points = 0.0
    for i in range(episodes):
        seat, dealer = i % 4, (i // 4) % 4
        agents = [baseline_factory(seed=seed * 7919 + i * 4 + k) for k in range(4)]  # seeded -> reproducible
        agents[seat] = agent
        res = play_hand(agents, dealer, seed=seed * 100003 + i)
        wins += res.winner == seat
        draws += res.kind == "draw"
        dealt_in += res.kind == "ron" and res.loser == seat
        points += res.deltas[seat]
    return {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "deal_in_rate": dealt_in / episodes,
        "avg_points": points / episodes,
    }


def append_csv(path: str, row: dict) -> None:
    """Append one row to a results CSV (creates it, with header, if needed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Evaluate an agent against baselines")
    ap.add_argument("--agent", default="heuristic", choices=["random", "heuristic"])
    ap.add_argument("--baseline", default="heuristic", choices=["random", "heuristic"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    agent = RandomAgent(a.seed) if a.agent == "random" else HeuristicAgent(a.seed)
    base = RandomAgent if a.baseline == "random" else HeuristicAgent
    t = time.time()
    print(evaluate(agent, base, a.episodes, a.seed), f"({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()

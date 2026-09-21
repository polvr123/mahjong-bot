import random

import numpy as np

from mahjong_rl.agents.heuristic_agent import HeuristicAgent
from mahjong_rl.agents.random_agent import RandomAgent
from mahjong_rl.engine import actions as A
from mahjong_rl.engine.game import MahjongGame
from mahjong_rl.tests.helpers import JUNK_A, JUNK_B, JUNK_C, TENPAI_3P, scripted_game
from mahjong_rl.training.evaluate import evaluate, play_hand


def test_agents_always_pick_legal_actions():
    for agent_cls in (RandomAgent, HeuristicAgent):
        for seed in range(20):
            g = MahjongGame(dealer=seed % 4, seed=seed)
            agent = agent_cls(seed)
            while not g.done:
                p = g.decision_player
                a = agent.act(g.state.player_view(p), g.legal_mask())
                assert g.legal_mask()[a]
                g.step(p, a)


def test_heuristic_takes_wins_and_riichi():
    g = scripted_game([JUNK_A, TENPAI_3P, JUNK_B, JUNK_C], mover=1, drawn=dict(pin="3"))
    g.state.riichi_declared[1] = True
    g.state.discards[1] = [0]
    assert HeuristicAgent(0).act(g.state.player_view(1), g.legal_mask()) == A.TSUMO

    g = scripted_game([dict(man="123456789", pin="99", sou="13"), TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    a = HeuristicAgent(0).act(g.state.player_view(0), g.legal_mask())
    assert A.RIICHI_BASE <= a < A.TSUMO


def test_heuristic_discards_the_obvious_tile():
    # 13 tiles are tenpai-ish except for one stray honor: the heuristic must discard the honor
    g = scripted_game([dict(man="123456789", pin="99", sou="13"), TENPAI_3P, JUNK_B, JUNK_C], drawn=dict(honors="7"))
    g.state.riichi_declared[0] = True  # forbid riichi so we exercise the plain-discard branch
    view = g.state.player_view(0)
    mask = np.zeros(A.NUM_ACTIONS, dtype=bool)
    mask[[A.DISCARD_BASE + 33, A.DISCARD_BASE + 18]] = True  # offer: discard red dragon or 1s
    assert HeuristicAgent(0).act(view, mask) == A.DISCARD_BASE + 33


def test_heuristic_beats_random():
    res = evaluate(HeuristicAgent(1), RandomAgent, episodes=200, seed=1)
    assert res["win_rate"] > 0.10  # random players essentially never win, so this is a large margin
    rand = evaluate(RandomAgent(1), RandomAgent, episodes=200, seed=2)
    assert res["win_rate"] > rand["win_rate"] + 0.10


def test_play_hand_returns_result():
    r = play_hand([RandomAgent(i) for i in range(4)], dealer=0, seed=3)
    assert sum(r.deltas) == 0

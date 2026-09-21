import numpy as np
import pytest

from mahjong_rl.agents.heuristic_agent import HeuristicAgent
from mahjong_rl.agents.random_agent import RandomAgent
from mahjong_rl.engine import actions as A
from mahjong_rl.engine.env import MahjongEnv
from mahjong_rl.engine.game import IllegalAction
from mahjong_rl.engine.observation import OBS_DIM


def run_episode(env, rng, seed=None, options=None):
    obs, info = env.reset(seed=seed, options=options)
    steps, total, done, last_info = 0, 0.0, False, {}
    while not done:
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
        mask = env.action_masks()
        assert mask.shape == (A.NUM_ACTIONS,) and mask.any()
        obs, r, done, trunc, last_info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert not trunc
        assert r == pytest.approx(last_info["raw_reward"] + last_info["shaped_reward"])
        total += r
        steps += 1
        assert steps < 500
    return steps, total, last_info


def test_random_episodes_terminate():
    env = MahjongEnv()
    rng = np.random.default_rng(0)
    outcomes = set()
    for i in range(60):
        steps, _, info = run_episode(env, rng, seed=i)
        assert info["outcome"] in ("win", "loss", "draw")
        outcomes.add(info["outcome"])
    assert "draw" in outcomes


def test_reset_options_pick_seat_and_dealer():
    env = MahjongEnv()
    for seat in range(4):
        _, info = env.reset(seed=seat, options={"seat": seat, "dealer": 3 - seat})
        assert info["agent_seat"] == seat and env.agent_seat == seat and info["dealer"] == 3 - seat


def test_illegal_action_is_rejected():
    env = MahjongEnv()
    env.reset(seed=0)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(IllegalAction):
        env.step(illegal)


def test_reward_is_zero_sum_outcome_plus_shaping():
    env = MahjongEnv(opponents=HeuristicAgent, shaping_coef=0.01)
    rng = np.random.default_rng(1)
    for i in range(30):
        _, _, info = run_episode(env, rng, seed=i)
        if info["outcome"] == "draw":
            assert info["raw_reward"] == 0.0
        # the shaping return telescopes to coef * (start shanten - last discard-time shanten)
        assert abs(info["episode_shaped_return"]) <= 0.01 * 8 + 1e-9


def test_shaping_can_be_switched_off():
    env = MahjongEnv(shaping_coef=0.0)
    rng = np.random.default_rng(2)
    for i in range(10):
        _, _, info = run_episode(env, rng, seed=i)
        assert info["episode_shaped_return"] == 0.0


def test_winner_gets_positive_raw_reward_and_losers_do_not():
    # a random learner never wins, so let the heuristic drive the learner seat
    env = MahjongEnv(opponents=HeuristicAgent, shaping_coef=0.0)
    driver = HeuristicAgent(0)
    seen = set()
    for i in range(150):
        obs, _ = env.reset(seed=i)
        done, total = False, 0.0
        while not done:
            obs, r, done, _, info = env.step(driver.act(env._view(), env.action_masks()))
            total += r
        if info["outcome"] == "win":
            assert total > 0 and info["points"] > 0
        elif info["outcome"] == "loss":
            assert total <= 0
        else:
            assert total == 0
        seen.add(info["outcome"])
    assert seen == {"win", "loss", "draw"}


def test_opponent_factory_is_called_per_seat_on_reset():
    made = []

    def factory():
        made.append(1)
        return RandomAgent()

    env = MahjongEnv(opponents=factory)
    env.reset(seed=0)
    assert len(made) == 4  # one per seat (the learner's slot is created but never used)
    env.reset(seed=1)
    assert len(made) == 8


def test_deterministic_given_seed():
    def trace(seed):
        env = MahjongEnv(opponents=lambda: RandomAgent(0))
        obs, _ = env.reset(seed=seed)
        return obs

    assert np.array_equal(trace(5), trace(5))

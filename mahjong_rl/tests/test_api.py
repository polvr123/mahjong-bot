import random

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from mahjong_rl.agents.heuristic_agent import HeuristicAgent
from mahjong_rl.engine import actions as A
from mahjong_rl.engine.env import MahjongEnv
from mahjong_rl.engine.game import MahjongGame
from mahjong_rl.serving.api import create_app
from mahjong_rl.serving.model_loader import load_agent


@pytest.fixture()
def client():
    with TestClient(create_app(agent=HeuristicAgent(0))) as c:
        yield c


def play_out(client, first, rng, opponent=None):
    state, steps = first, 0
    while not state["done"]:
        assert state["legal"], "human must always have a legal action when the hand is live"
        state = client.post(f"/game/{state['game_id']}/action", json={"action": rng.choice(state["legal"])}).json()
        steps += 1
        assert steps < 200
    return state


def test_health_and_index(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "mahjong" in r.text and "/game/new" in r.text


@pytest.mark.parametrize("opponent", ["ppo", "heuristic", "random"])
def test_full_hand_over_http(client, opponent):
    rng = random.Random(0)
    for seat in range(4):
        first = client.post("/game/new", json={"seat": seat, "dealer": (seat + 1) % 4, "opponent": opponent}).json()
        assert first["you"] == seat and len(first["hand"]) == 13 or first["done"]
        final = play_out(client, first, rng)
        assert final["done"] and sum(final["result"]["deltas"]) == 0
        assert len(final["final_hands"]) == 4
        # game is finished: further actions are rejected
        r = client.post(f"/game/{final['game_id']}/action", json={"action": 0})
        assert r.status_code == 400


def test_human_never_sees_hidden_tiles(client):
    s = client.post("/game/new", json={"seat": 1}).json()
    assert "final_hands" in s and s["final_hands"] is None or s["done"]
    assert "hands" not in s and "wall" not in s  # opponents' hands and the wall are not in a live state
    assert len(s["hand"]) + (s["drawn_tile"] is not None) in (13, 14)


def test_illegal_and_unknown(client):
    s = client.post("/game/new", json={"seat": 0, "dealer": 0}).json()
    illegal = next(a for a in range(A.NUM_ACTIONS) if a not in s["legal"])
    assert client.post(f"/game/{s['game_id']}/action", json={"action": illegal}).status_code == 400
    assert client.post(f"/game/{s['game_id']}/action", json={"action": 999}).status_code == 400
    assert client.post("/game/nope/action", json={"action": 0}).status_code == 404
    assert client.get("/game/nope").status_code == 404
    assert client.post("/game/new", json={"opponent": "alien"}).status_code == 400
    assert client.post("/game/new", json={"seat": 9}).status_code == 400


def test_stateless_move_endpoint(client):
    g = MahjongGame(dealer=0, seed=4)
    p = g.decision_player
    r = client.post("/move", json={"view": g.state.player_view(p), "legal_actions": g.legal_actions()})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in g.legal_actions() and isinstance(body["name"], str)
    assert client.post("/move", json={"view": g.state.player_view(p), "legal_actions": []}).status_code == 400
    assert client.post("/move", json={"view": {}, "legal_actions": [0]}).status_code == 400


def test_loader_falls_back_without_a_model(tmp_path):
    agent, desc = load_agent(str(tmp_path / "missing.zip"))
    assert isinstance(agent, HeuristicAgent) and "no trained model" in desc


def test_real_ppo_checkpoint_serves_moves(tmp_path):
    env = ActionMasker(MahjongEnv(), lambda e: e.unwrapped.action_masks())
    path = str(tmp_path / "tiny.zip")
    MaskablePPO("MlpPolicy", env, n_steps=32, batch_size=32, policy_kwargs=dict(net_arch=[16]), device="cpu").save(path)
    with TestClient(create_app(model_path=path)) as c:
        assert c.get("/health").json()["model"] == "ppo:tiny.zip"
        g = MahjongGame(dealer=2, seed=9)
        p = g.decision_player
        r = c.post("/move", json={"view": g.state.player_view(p), "legal_actions": g.legal_actions()})
        assert r.json()["action"] in g.legal_actions()
        first = c.post("/game/new", json={"opponent": "ppo"}).json()
        assert play_out(c, first, random.Random(1))["done"]

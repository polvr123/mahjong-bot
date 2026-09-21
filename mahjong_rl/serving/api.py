"""FastAPI app: serves the web page, a stateless /move endpoint, and in-memory single-hand games.

    python -m uvicorn mahjong_rl.serving.api:app --port 8000

Endpoints
    GET  /                    the web page
    GET  /health              which model is loaded
    POST /move                stateless: {view, legal_actions} -> the bot's chosen action
    POST /game/new            start a hand for a human seat vs 3 bots -> game state
    POST /game/{id}/action    human acts, bots play until the human must act again -> game state
    GET  /game/{id}           current game state
"""
from __future__ import annotations

import os
import random
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..agents.heuristic_agent import HeuristicAgent
from ..agents.random_agent import RandomAgent
from ..engine import actions as A
from ..engine.game import IllegalAction, MahjongGame
from .model_loader import load_agent

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
MAX_SESSIONS = 500
REL_NAMES = ["You", "Right", "Across", "Left"]  # seat relative to the human, in turn order


class MoveRequest(BaseModel):
    view: dict          # GameState.player_view(seat) as JSON
    legal_actions: list[int]


class NewGameRequest(BaseModel):
    seat: int | None = None       # human seat (default random)
    dealer: int | None = None     # dealer seat (default random)
    opponent: str = "ppo"         # "ppo" | "heuristic" | "random"


class ActionRequest(BaseModel):
    action: int


class Session:
    def __init__(self, game: MahjongGame, seat: int, bots: list, opponent: str):
        self.game, self.seat, self.bots, self.opponent = game, seat, bots, opponent
        self.events: list[str] = []

    def who(self, p: int) -> str:
        return REL_NAMES[(p - self.seat) % 4]

    def describe(self, p: int, a: int) -> str:
        if a == A.TSUMO:
            return f"{self.who(p)} declares tsumo"
        if a == A.RON:
            return f"{self.who(p)} declares ron"
        if a == A.PASS:
            return f"{self.who(p)} passes on the ron"
        if a >= A.RIICHI_BASE:
            return f"{self.who(p)} declares riichi, discarding {A.action_name(a).split()[-1]}"
        return f"{self.who(p)} discards {A.action_name(a).split()[-1]}"

    def apply(self, p: int, a: int) -> None:
        self.events.append(self.describe(p, a))
        self.game.step(p, a)

    def advance_bots(self) -> None:
        g = self.game
        while not g.done and g.decision_player != self.seat:
            p = g.decision_player
            self.apply(p, self.bots[p].act(g.state.player_view(p), g.legal_mask()))

    def snapshot(self, events: list[str]) -> dict:
        g, s = self.game, self.game.state
        view = s.player_view(self.seat)
        legal = g.legal_actions() if (not g.done and g.decision_player == self.seat) else []
        out = {
            "you": self.seat,
            "dealer": s.dealer,
            "seat_winds": s.seat_winds,
            "round_wind": s.round_wind,
            "hand": view["hand"],
            "drawn_tile": view["drawn_tile"],
            "discards": view["discards"],
            "dora_indicators": view["dora_indicators"],
            "riichi": view["riichi"],
            "wall_remaining": view["wall_remaining"],
            "phase": view["phase"],
            "pending_tile": view["pending_tile"],
            "pending_from": view["pending_from"],
            "furiten": view["furiten"],
            "turn": s.decision_player if not g.done else None,
            "legal": legal,
            "legal_names": {str(a): A.action_name(a) for a in legal},
            "events": events,
            "opponent": self.opponent,
            "done": g.done,
            "result": None,
            "final_hands": None,
        }
        if g.done:
            r = g.result
            out["result"] = {
                "kind": r.kind, "winner": r.winner, "loser": r.loser, "han": r.han, "fu": r.fu,
                "points": r.points, "yaku": r.yaku, "deltas": r.deltas,
            }
            out["final_hands"] = [sorted(h) for h in s.hands]
        return out


def _make_bots(opponent: str, ppo_agent, seed: int) -> list:
    if opponent == "ppo":
        return [ppo_agent] * 4
    if opponent == "heuristic":
        return [HeuristicAgent(seed + i) for i in range(4)]
    if opponent == "random":
        return [RandomAgent(seed + i) for i in range(4)]
    raise HTTPException(400, f"unknown opponent '{opponent}' (use ppo, heuristic or random)")


def create_app(agent=None, model_path: str | None = None) -> FastAPI:
    """`agent` overrides model loading (used by tests). Otherwise the checkpoint loads once at startup."""
    state: dict = {"agent": agent, "model": "custom" if agent else None}
    sessions: OrderedDict[str, Session] = OrderedDict()
    lock = threading.Lock()  # one shared model + in-memory sessions: keep it simple and serialise requests

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["agent"] is None:
            state["agent"], state["model"] = load_agent(model_path)
        yield

    app = FastAPI(title="Riichi Mahjong RL demo", lifespan=lifespan)

    def get_session(game_id: str) -> Session:
        sess = sessions.get(game_id)
        if sess is None:
            raise HTTPException(404, "unknown game id (server restarted or game expired)")
        return sess

    @app.get("/health")
    def health():
        return {"status": "ok", "model": state["model"], "active_games": len(sessions)}

    @app.post("/move")
    def move(req: MoveRequest):
        """Stateless: the client sends its view and the legal action ids; the model picks one."""
        if not req.legal_actions or any(not 0 <= a < A.NUM_ACTIONS for a in req.legal_actions):
            raise HTTPException(400, "legal_actions must be a non-empty list of action ids in [0, 70]")
        mask = np.zeros(A.NUM_ACTIONS, dtype=bool)
        mask[req.legal_actions] = True
        try:
            with lock:
                action = state["agent"].act(req.view, mask)
        except (KeyError, TypeError, IndexError, ValueError) as e:
            raise HTTPException(400, f"malformed view: {e!r}")
        return {"action": action, "name": A.action_name(action)}

    @app.post("/game/new")
    def new_game(req: NewGameRequest | None = None):
        req = req or NewGameRequest()
        rng = random.Random()
        seat = rng.randrange(4) if req.seat is None else req.seat
        dealer = rng.randrange(4) if req.dealer is None else req.dealer
        if not (0 <= seat < 4 and 0 <= dealer < 4):
            raise HTTPException(400, "seat and dealer must be 0-3")
        with lock:
            bots = _make_bots(req.opponent, state["agent"], rng.randrange(2**31))
            sess = Session(MahjongGame(dealer=dealer, rng=rng), seat, bots, req.opponent)
            sess.advance_bots()
            game_id = uuid.uuid4().hex[:12]
            sessions[game_id] = sess
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
            events, sess.events = sess.events, []
            return {"game_id": game_id, **sess.snapshot(events)}

    @app.get("/game/{game_id}")
    def get_game(game_id: str):
        with lock:
            return {"game_id": game_id, **get_session(game_id).snapshot([])}

    @app.post("/game/{game_id}/action")
    def game_action(game_id: str, req: ActionRequest):
        with lock:
            sess = get_session(game_id)
            g = sess.game
            if g.done:
                raise HTTPException(400, "this hand is over; start a new one")
            if not (0 <= req.action < A.NUM_ACTIONS) or not g.legal_mask()[req.action]:
                raise HTTPException(400, f"illegal action {req.action}; legal: {g.legal_actions()}")
            try:
                sess.apply(sess.seat, req.action)
                sess.advance_bots()
            except IllegalAction as e:  # should be unreachable given the checks above
                raise HTTPException(400, str(e))
            events, sess.events = sess.events, []
            sessions.move_to_end(game_id)
            return {"game_id": game_id, **sess.snapshot(events)}

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    return app


app = create_app()

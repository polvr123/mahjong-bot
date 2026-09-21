"""Load the trained agent once at server startup."""
from __future__ import annotations

import logging
import os

from ..agents.heuristic_agent import HeuristicAgent
from ..agents.ppo_agent import PPOAgent

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "model.zip")


def load_agent(path: str | None = None):
    """Return (agent, description). Path order: argument, $MAHJONG_MODEL, serving/models/model.zip.
    If no checkpoint exists the heuristic bot is used so the demo still runs (and says so)."""
    path = path or os.environ.get("MAHJONG_MODEL") or DEFAULT_MODEL
    if os.path.exists(path):
        log.info("loading model %s", path)
        return PPOAgent.load(path, deterministic=True), f"ppo:{os.path.basename(path)}"
    log.warning("no model at %s; falling back to the heuristic bot", path)
    return HeuristicAgent(), "heuristic (no trained model found)"

"""Thin wrapper turning a trained MaskablePPO policy into an Agent."""
from __future__ import annotations

import numpy as np

from ..engine.observation import encode_view


class PPOAgent:
    """Agent backed by a MaskablePPO policy. `deterministic=True` picks the argmax action
    (evaluation / serving); False samples from the policy (more varied opponents in self-play)."""

    name = "ppo"

    def __init__(self, policy, deterministic: bool = False):
        self.policy = policy
        self.deterministic = deterministic
        self.policy.set_training_mode(False)

    @classmethod
    def load(cls, path: str, deterministic: bool = True, device: str = "cpu") -> "PPOAgent":
        from sb3_contrib import MaskablePPO

        return cls(MaskablePPO.load(path, device=device).policy, deterministic)

    @classmethod
    def snapshot(cls, policy, deterministic: bool = False) -> "PPOAgent":
        """Frozen copy of a policy that is still being trained.

        Copies the weights into a freshly built policy rather than deep-copying the object: right after
        a PPO update the policy caches its last action distribution, which still carries a gradient graph
        and makes copy.deepcopy raise."""
        clone = policy.__class__(**policy._get_constructor_parameters())
        clone.load_state_dict(policy.state_dict())
        clone.to(policy.device)
        return cls(clone, deterministic)

    def act(self, view: dict, mask: np.ndarray) -> int:
        action, _ = self.policy.predict(encode_view(view), deterministic=self.deterministic, action_masks=mask)
        return int(np.asarray(action).reshape(-1)[0])

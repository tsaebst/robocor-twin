from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from env import RCConfig, make_env


class RoboCourierTwin:
    """
    Digital twin of RoboCourier: parameterized by theta via RCConfig.
    """
    def __init__(self, theta: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        self.theta = theta or {}
        self.seed = seed
        self.env = self._build_env()

        self.action_space = self.env.action_space
        self.observation_space = getattr(self.env, "observation_space", None)

    def _build_env(self):
        base = RCConfig()
        cfg = asdict(base)

        # Apply theta (only known keys)
        cfg.update(self.theta)

        # Optionally override seed for reproducibility
        if self.seed is not None:
            cfg["seed"] = self.seed

        return make_env(RCConfig(**cfg))

    def set_theta(self, theta: Dict[str, Any]):
        self.theta = dict(theta)
        self.env = self._build_env()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


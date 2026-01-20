from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

from rc_calib.init_state import force_world_state

class FixedInitTwinWrapper:
    """
    Wraps a RoboCourierEnv-like env and forces its initial world state on reset().
    """
    def __init__(self, env, init_state: Dict[str, Any]):
        self.env = env
        self.init_state = init_state

        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # force identical initial state
        force_world_state(self.env, self.init_state)

        # recompute observation from forced state
        if hasattr(self.env, "_obs") and callable(getattr(self.env, "_obs")):
            obs = self.env._obs()
        else:
            # fallback: trust obs if _obs is not accessible
            obs = obs

        return np.array(obs, dtype=np.float32), info

    def step(self, action: Any):
        return self.env.step(action)


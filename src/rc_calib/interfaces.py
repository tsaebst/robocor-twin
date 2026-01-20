from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

Theta = Dict[str, Any]

@dataclass
class OracleSample:
    t_global: int
    episode_id: int
    ep_t: int
    action: int
    obs_oracle: np.ndarray
    reward_oracle: float
    world_state: Dict[str, Any]

@dataclass
class TwinSample:
    t_global: int
    episode_id: int
    ep_t: int
    action: int
    obs_twin: np.ndarray
    reward_twin: float

class Policy(Protocol):
    name: str

    def reset(self, *, episode_id: int, seed: int) -> None:
        ...

    def act(self, obs_twin: np.ndarray, theta: Theta) -> int:
        ...

    def wants_oracle(self, obs_twin: np.ndarray, theta: Theta, t_global: int, ep_t: int) -> bool:
        """Policy can request oracle, but runner enforces budget."""
        ...

class Calibrator(Protocol):
    name: str

    def reset(self, theta0: Theta) -> None:
        ...

    def observe(self, twin: TwinSample, oracle: OracleSample) -> None:
        ...

    def ready_to_update(self) -> bool:
        ...

    def update(self, theta_current: Theta) -> Tuple[Theta, Dict[str, float]]:
        """Return (new_theta, metrics_summary)."""
        ...


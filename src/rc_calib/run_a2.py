import numpy as np
from typing import Dict, Any
from rc_calib.interfaces import Policy, Theta

class A2Policy:
    name = "A2_greedy_charge"

    def __init__(self, charge_thresh: float = 0.30, act_map: Dict[str, int] = None):
        self.charge_thresh = float(charge_thresh)
        self.act_map = act_map or {}

    def reset(self, *, episode_id: int, seed: int) -> None:
        pass

    def wants_oracle(self, obs_twin: np.ndarray, theta: Theta, t_global: int, ep_t: int) -> bool:
        # usually False; runner already queries periodically
        return False

    def act(self, obs_twin: np.ndarray, theta: Theta) -> int:
        grid_size = int(theta["grid_size"])
        # decode positions (same indices as before)
        rx, ry = obs_twin[0], obs_twin[1]
        px, py = obs_twin[2], obs_twin[3]
        dx, dy = obs_twin[4], obs_twin[5]
        has = float(obs_twin[6]) > 0.5
        bat = float(obs_twin[7])
        cx, cy = obs_twin[8], obs_twin[9]

        # choose normalized target
        if bat <= self.charge_thresh:
            tx, ty = cx, cy
        else:
            tx, ty = (dx, dy) if has else (px, py)

        # greedy in normalized space (works if actions move on grid deterministically)
        if rx < tx: return self.act_map["right"]
        if rx > tx: return self.act_map["left"]
        if ry < ty: return self.act_map["up"]
        if ry > ty: return self.act_map["down"]
        return self.act_map["up"]


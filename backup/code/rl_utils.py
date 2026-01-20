from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import requests

from env import RCConfig, make_env


# ============================================================
# Oracle client (HTTP)
# ============================================================

class RoboCourierOracle:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def reset(self, seed: int = 0, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"seed": seed, "config_overrides": config_overrides}
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, action: int) -> Dict[str, Any]:
        payload = {"session_id": session_id, "action": int(action)}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


# ============================================================
# JSON safety (avoid NaN in logs)
# ============================================================

def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None to keep JSON valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj

def json_dumps_safe(obj: Any) -> str:
    """Safe dumps that never emits NaN."""
    return json.dumps(_sanitize(obj), ensure_ascii=False, allow_nan=False)


# ============================================================
# Helpers: coords + action map
# ============================================================

def norm_xy_to_grid(xn: float, yn: float, grid_size: int) -> Tuple[int, int]:
    gs = grid_size - 1
    x = int(round(float(xn) * gs))
    y = int(round(float(yn) * gs))
    x = max(0, min(gs, x))
    y = max(0, min(gs, y))
    return x, y

def obs_to_robot_xy(obs: np.ndarray, grid_size: int) -> Tuple[int, int]:
    return norm_xy_to_grid(float(obs[0]), float(obs[1]), grid_size)

def greedy_action_towards(curr: Tuple[int, int], target: Tuple[int, int], act: Dict[str, int]) -> int:
    cx, cy = curr
    tx, ty = target
    if cx < tx:
        return act["right"]
    if cx > tx:
        return act["left"]
    if cy < ty:
        return act["up"]
    if cy > ty:
        return act["down"]
    return act["up"]

def infer_action_mapping_local(seed: int = 0) -> Dict[str, int]:
    e = make_env(RCConfig(seed=seed))
    obs0, _ = e.reset(seed=seed)
    x0, y0 = obs_to_robot_xy(np.asarray(obs0), e.grid_size)

    mapping: Dict[str, int] = {}
    for a in range(e.action_space.n):
        e.reset(seed=seed)
        obs1, _, _, _, _ = e.step(a)
        x1, y1 = obs_to_robot_xy(np.asarray(obs1), e.grid_size)
        dx, dy = x1 - x0, y1 - y0
        if (dx, dy) == (0, 1):
            mapping["up"] = a
        elif (dx, dy) == (1, 0):
            mapping["right"] = a
        elif (dx, dy) == (0, -1):
            mapping["down"] = a
        elif (dx, dy) == (-1, 0):
            mapping["left"] = a

    if len(mapping) < 4:
        raise RuntimeError(f"Could not infer full action map, got: {mapping}")
    return mapping


# ============================================================
# Theta handling
# ============================================================

@dataclass
class Theta:
    grid_size: int = 10
    use_stay: bool = False
    battery_max: int = 80
    step_cost: float = 0.10
    delivery_reward: float = 10.0
    battery_fail_penalty: float = 8.0
    p_slip: float = 0.10

def theta_to_dict(th: Theta) -> Dict[str, Any]:
    return {
        "grid_size": int(th.grid_size),
        "use_stay": bool(th.use_stay),
        "battery_max": int(th.battery_max),
        "step_cost": float(th.step_cost),
        "delivery_reward": float(th.delivery_reward),
        "battery_fail_penalty": float(th.battery_fail_penalty),
        "p_slip": float(th.p_slip),
    }

def clamp_theta(th: Theta) -> Theta:
    th.battery_max = int(np.clip(th.battery_max, 30, 400))
    th.step_cost = float(np.clip(th.step_cost, 0.01, 0.50))
    th.delivery_reward = float(np.clip(th.delivery_reward, 1.0, 20.0))
    th.battery_fail_penalty = float(np.clip(th.battery_fail_penalty, 0.0, 25.0))
    th.p_slip = float(np.clip(th.p_slip, 0.0, 0.30))
    return th

def ema_update(curr: float, new: float, alpha: float) -> float:
    return (1.0 - float(alpha)) * float(curr) + float(alpha) * float(new)

def unwrap_env(env_obj: Any, max_depth: int = 20) -> Any:
    base = env_obj
    for _ in range(max_depth):
        if hasattr(base, "env"):
            base = getattr(base, "env")
        else:
            break
    return base

def apply_theta_to_env(env_obj: Any, th: Theta) -> None:
    base = unwrap_env(env_obj)

    for k in ["grid_size", "battery_max", "step_cost", "delivery_reward", "battery_fail_penalty", "use_stay"]:
        if hasattr(base, k):
            setattr(base, k, getattr(th, k))

    w = env_obj
    for _ in range(20):
        if hasattr(w, "p_slip"):
            setattr(w, "p_slip", float(th.p_slip))
            break
        if hasattr(w, "env"):
            w = w.env
        else:
            break


# ============================================================
# Plot helpers (optional - keep if you want)
# ============================================================

def compress_to_arrows(x: np.ndarray, y: np.ndarray):
    arrows = []
    if len(x) < 2:
        return arrows
    dxs = np.diff(x)
    dys = np.diff(y)
    i = 0
    while i < len(dxs):
        if dxs[i] == 0 and dys[i] == 0:
            i += 1
            continue
        x0, y0 = x[i], y[i]
        dx_acc, dy_acc = dxs[i], dys[i]
        j = i + 1
        while j < len(dxs) and dxs[j] == dxs[i] and dys[j] == dys[i]:
            dx_acc += dxs[j]
            dy_acc += dys[j]
            j += 1
        arrows.append((float(x0), float(y0), float(dx_acc), float(dy_acc)))
        i = j
    return arrows

def plot_arrows(ax, arrows, color: str, label: str):
    if not arrows:
        return
    x0 = np.array([a[0] for a in arrows], dtype=float)
    y0 = np.array([a[1] for a in arrows], dtype=float)
    dx = np.array([a[2] for a in arrows], dtype=float)
    dy = np.array([a[3] for a in arrows], dtype=float)
    ax.quiver(x0, y0, dx, dy, angles="xy", scale_units="xy", scale=1.0, color=color, label=label)

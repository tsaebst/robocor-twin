# calib/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
import numpy as np

EPS = 1e-9

def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)

def huber(x: np.ndarray, delta: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    quad = np.minimum(ax, delta)
    lin = ax - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))

def bernoulli_nll(p: float, y: np.ndarray) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def masked_theta_errors(theta_est: Dict[str, float], theta_true: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v_true in (theta_true or {}).items():
        if v_true is None or is_nan(v_true):
            continue
        if k not in theta_est:
            continue
        out[f"abs_err/{k}"] = abs(float(theta_est[k]) - float(v_true))
    return out

@dataclass
class CalibBatch:
    # shape [T,2] / [T]
    pos_next_oracle: np.ndarray
    pos_next_twin: np.ndarray
    bat_next_oracle: np.ndarray
    bat_next_twin: np.ndarray
    reward_oracle: np.ndarray
    reward_twin: np.ndarray
    done_oracle: np.ndarray
    done_twin: np.ndarray
    slip_flag: Optional[np.ndarray] = None  # [T] 0/1

def compute_loss_components(
    batch: CalibBatch,
    theta_est: Dict[str, float],
    weights: Dict[str, float],
    huber_delta: float = 1.0,
) -> Dict[str, float]:
    l_pos = huber(batch.pos_next_twin - batch.pos_next_oracle, delta=huber_delta)
    l_bat = huber((batch.bat_next_twin - batch.bat_next_oracle).reshape(-1, 1), delta=huber_delta)
    l_rew = huber((batch.reward_twin - batch.reward_oracle).reshape(-1, 1), delta=huber_delta)

    done_t = np.clip(batch.done_twin.astype(float), 0.0, 1.0)
    done_o = np.clip(batch.done_oracle.astype(float), 0.0, 1.0)
    l_term = float(np.mean(np.abs(done_t - done_o)))

    if batch.slip_flag is not None and "p_slip" in theta_est:
        l_slip = bernoulli_nll(float(theta_est["p_slip"]), batch.slip_flag.astype(float))
    else:
        l_slip = 0.0

    total = (
        weights.get("pos", 1.0) * l_pos
        + weights.get("bat", 1.0) * l_bat
        + weights.get("rew", 1.0) * l_rew
        + weights.get("term", 1.0) * l_term
        + weights.get("slip", 1.0) * l_slip
    )
    return {
        "loss/pos": float(l_pos),
        "loss/bat": float(l_bat),
        "loss/rew": float(l_rew),
        "loss/term": float(l_term),
        "loss/slip": float(l_slip),
        "loss/total": float(total),
    }

def compute_accuracy_metrics(
    batch: CalibBatch,
    theta_est: Dict[str, float],
    theta_true: Dict[str, float],
) -> Dict[str, float]:
    pos_mse = float(np.mean(np.sum((batch.pos_next_twin - batch.pos_next_oracle) ** 2, axis=1)))
    bat_mse = float(np.mean((batch.bat_next_twin - batch.bat_next_oracle) ** 2))
    rew_mae = float(np.mean(np.abs(batch.reward_twin - batch.reward_oracle)))

    theta_abs = masked_theta_errors(theta_est, theta_true)
    theta_mean_abs = float(np.mean(list(theta_abs.values()))) if theta_abs else float("nan")

    out: Dict[str, float] = {
        "acc/pos_mse": pos_mse,
        "acc/bat_mse": bat_mse,
        "acc/rew_mae": rew_mae,
        "acc/theta_mean_abs": theta_mean_abs,
    }
    out.update(theta_abs)
    return out

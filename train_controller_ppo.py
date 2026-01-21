from __future__ import annotations

import gymnasium as gym
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from env import RCConfig, make_env
from src.rc_calib.wrappers import SlipActionWrapper
from metrics import CalibBatch, compute_loss_components, compute_accuracy_metrics
from rl_utils import (
    RoboCourierOracle,
    Theta,
    clamp_theta,
    theta_to_dict,
    infer_action_mapping_local,
    obs_to_robot_xy,
    norm_xy_to_grid,
    greedy_action_towards,
    apply_theta_to_env,
)

# ============================================================
# NO hardcoding / NO GT leakage:
# - theta_true is EVAL/LOG ONLY.
# - policy observation/reward/cost use ONLY disagreement metrics:
#   loss_parts + (pos_mse, bat_mse, rew_mae) computed WITHOUT theta_true.
#
# Stability upgrades:
# - online normalization (RunningMeanStd) for last_signals.
# - annealed prior reg + annealed update scales.
# - smoothness penalties for all calibrated params.
# - generic bounds barrier (prevents sticking to clamp edges).
# ============================================================

LOSS_WEIGHTS = {
    "pos": 1.0,
    "bat": 2.0,
    "rew": 0.2,   # keep low (rew easiest to soak)
    "term": 1.0,
    "slip": 0.5,
    "param_reg": 0.0,
}
CALIB_BATCH_LEN = 32
HUBER_DELTA = 1.0

# Normalization scales (NOT GT; just stable units for obs/penalties)
NORM_BATTERY_MAX = 400.0
NORM_STEP_COST = 0.50
NORM_P_SLIP = 0.30

# For bounds barrier: normalized margin near edges [0,1]
DEFAULT_BOUNDS_EPS = 0.08


def a5_driver_action(obs: np.ndarray, grid_size: int, act_map: Dict[str, int], charge_thresh: float = 0.30) -> int:
    has = float(obs[6]) > 0.5
    bat = float(obs[7])
    rx, ry = obs_to_robot_xy(obs, grid_size)

    px, py = norm_xy_to_grid(float(obs[2]), float(obs[3]), grid_size)
    dx, dy = norm_xy_to_grid(float(obs[4]), float(obs[5]), grid_size)
    cx, cy = norm_xy_to_grid(float(obs[8]), float(obs[9]), grid_size)

    if bat <= float(charge_thresh):
        target = (cx, cy)
    else:
        target = (dx, dy) if has else (px, py)
    return greedy_action_towards((rx, ry), target, act_map)


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def json_dumps_safe(obj: Any) -> str:
    return json.dumps(_sanitize_json(obj), ensure_ascii=False)


def _is_valid_gt(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return True


@dataclass
class PairTransition:
    twin_pos_next: np.ndarray
    oracle_pos_next: np.ndarray
    twin_bat_next: float
    oracle_bat_next: float
    twin_rew: float
    oracle_rew: float
    twin_done: int
    oracle_done: int


class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf: List[PairTransition] = []

    def append(self, x: PairTransition) -> None:
        self._buf.append(x)
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def tail(self, n: int) -> List[PairTransition]:
        n = int(n)
        if n <= 0:
            return []
        return self._buf[-n:]

    def __len__(self) -> int:
        return len(self._buf)


class RunningMeanStd:
    """
    Online mean/std (Welford) for vector signals.
    Used to normalize last_signals fed to the policy.
    """
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-8):
        self.shape = tuple(shape)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.M2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.shape:
            raise ValueError(f"RMS shape mismatch: got {x.shape}, expected {self.shape}")
        self.n += 1
        delta = x - self.mean
        self.mean += delta / float(self.n)
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.shape, dtype=np.float64)
        var = self.M2 / float(self.n - 1)
        return np.sqrt(np.maximum(var, 0.0) + self.eps)

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / self.std()
        z = np.clip(z, -float(clip), float(clip))
        return z.astype(np.float32)


class ControllerEnv(gym.Env):
    """
    PPO action:
      action[0] = query gate
      action[1] = delta battery_max
      action[2] = delta step_cost
      action[3] = delta p_slip

    NO GT leakage:
      - theta_true is logged only.
      - policy uses ONLY loss_parts + acc parts computed with EMPTY theta_true.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        oracle_url: str,
        driver_model_path: Optional[str],
        log_path: str,
        seed0: int = 0,
        H: int = 1000,
        K: int = 200,
        query_every: int = 20,
        sleep_s: float = 0.0,
        query_cost_base: float = 0.10,
        step_penalty: float = 0.001,
        # prior regularization (annealed)
        theta_reg: float = 0.25,
        theta_reg_final_frac: float = 0.10,
        theta_reg_battery_mult: float = 0.25,
        theta_reg_step_mult: float = 1.00,
        theta_reg_pslip_mult: float = 1.00,
        charge_thresh: float = 0.30,
        init_theta: Optional[Theta] = None,
        prior_theta: Optional[Theta] = None,
        curriculum_steps: int = 50_000,
        curriculum_stages: Optional[List[Dict[str, Any]]] = None,
        # update scales (annealed)
        delta_battery_scale: float = 12.0,
        delta_step_cost_scale: float = 0.010,
        delta_pslip_scale: float = 0.010,
        delta_min_frac: float = 0.20,
        delta_anneal_power: float = 1.0,
        # smoothness penalties (normalized)
        battery_smooth_w: float = 0.30,
        step_cost_smooth_w: float = 0.50,
        pslip_smooth_w: float = 0.20,
        # bounds barrier
        bounds_barrier_w: float = 0.25,
        bounds_barrier_eps: float = DEFAULT_BOUNDS_EPS,
        # query budget schedule
        k_min_late: int = 10,
        k_decay_power: float = 2.0,
        # signal normalization
        normalize_signals: bool = True,
        signals_clip: float = 5.0,
        # query cost shaping
        query_time_shaping: bool = True,   # if False => no exp(3p); mostly error-based
        query_time_exp_k: float = 3.0,     # exp(k * p) if enabled
    ):
        super().__init__()
        self.oracle = RoboCourierOracle(oracle_url)
        self.seed0 = int(seed0)
        self._rng = np.random.default_rng(self.seed0)

        self.H = int(H)
        self.K = int(K)
        self.query_every_base = int(query_every)
        self.sleep_s = float(sleep_s)

        self.query_cost_base = float(query_cost_base)
        self.step_penalty = float(step_penalty)
        self.charge_thresh = float(charge_thresh)

        self.loss_weights = dict(LOSS_WEIGHTS)
        self.huber_delta = float(HUBER_DELTA)
        self.batch_len = int(CALIB_BATCH_LEN)
        self.min_T = max(8, self.batch_len // 2)

        self.curriculum_steps = int(curriculum_steps)

        self.theta_reg = float(theta_reg)
        self.theta_reg_final_frac = float(theta_reg_final_frac)
        self.theta_reg_battery_mult = float(theta_reg_battery_mult)
        self.theta_reg_step_mult = float(theta_reg_step_mult)
        self.theta_reg_pslip_mult = float(theta_reg_pslip_mult)

        self.delta_battery_scale0 = float(delta_battery_scale)
        self.delta_step_cost_scale0 = float(delta_step_cost_scale)
        self.delta_pslip_scale0 = float(delta_pslip_scale)
        self.delta_min_frac = float(delta_min_frac)
        self.delta_anneal_power = float(delta_anneal_power)

        self.battery_smooth_w = float(battery_smooth_w)
        self.step_cost_smooth_w = float(step_cost_smooth_w)
        self.pslip_smooth_w = float(pslip_smooth_w)
        self._prev_battery_for_smooth: Optional[float] = None
        self._prev_step_cost_for_smooth: Optional[float] = None
        self._prev_pslip_for_smooth: Optional[float] = None

        self.bounds_barrier_w = float(bounds_barrier_w)
        self.bounds_barrier_eps = float(bounds_barrier_eps)

        self.k_min_late = int(k_min_late)
        self.k_decay_power = float(k_decay_power)
        self.K_eff = int(self.K)

        self.normalize_signals = bool(normalize_signals)
        self.signals_clip = float(signals_clip)
        self._signals_rms = RunningMeanStd(shape=(8,))

        self.query_time_shaping = bool(query_time_shaping)
        self.query_time_exp_k = float(query_time_exp_k)

        if curriculum_stages is None:
            self.curriculum_stages = [
                {
                    "name": "stage0_easy_no_slip",
                    "until_frac": 0.30,
                    "slip_enabled": False,
                    "allow_pslip_update": False,
                    "query_every": self.query_every_base,
                },
                {
                    "name": "stage1_slip_on_pslip_frozen",
                    "until_frac": 0.70,
                    "slip_enabled": True,
                    "allow_pslip_update": False,
                    "query_every": max(10, self.query_every_base),
                },
                {
                    "name": "stage2_full",
                    "until_frac": 1.01,
                    "slip_enabled": True,
                    "allow_pslip_update": True,
                    "query_every": max(20, self.query_every_base),
                },
            ]
        else:
            self.curriculum_stages = curriculum_stages

        # Logging
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_f = self.log_path.open("w", encoding="utf-8")

        # action mapping
        self.act_map = infer_action_mapping_local(seed=self.seed0)

        # driver model
        self.driver_is_heuristic = (driver_model_path is None)
        self.driver_model = PPO.load(driver_model_path) if (not self.driver_is_heuristic) else None

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: 10 + 3 + 1 + 1 + 8 + 1 + 1 = 25
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(10 + 3 + 1 + 1 + 8 + 1 + 1,), dtype=np.float32
        )

        self.theta = init_theta if init_theta is not None else Theta()
        self.theta_prior = prior_theta if prior_theta is not None else Theta(
            battery_max=80, step_cost=0.10, delivery_reward=10.0, battery_fail_penalty=8.0, p_slip=0.10
        )

        # runtime state
        self.sid: Optional[str] = None
        self.grid_size: int = 10

        self.ep = 0
        self.t_ep = 0
        self.t_global = 0

        self.k_used_ep = 0
        self.k_used_total = 0
        self.since_query = 10**9
        self.update_idx = 0

        # last_signals: [pos_mse, bat_mse, rew_mae, loss_pos, loss_bat, loss_rew, loss_term, loss_total]
        self.last_signals = np.zeros((8,), dtype=np.float32)

        self.obs_t: Optional[np.ndarray] = None
        self.twin = None

        # GT: EVAL/LOG ONLY
        self.theta_true: Dict[str, Any] = {}
        self.gt_has_pslip: bool = False

        self.pair_buf = RingBuffer(capacity=max(256, self.batch_len * 4))

        # curriculum cache
        self._curr_stage_idx = 0
        self._curr_stage_name = "unknown"
        self._curr_slip_enabled = True
        self._curr_allow_pslip_update = True
        self._curr_query_every = self.query_every_base

        self.prev_loss_total: Optional[float] = None

    # ---------------- schedules ----------------
    def _progress_frac(self) -> float:
        denom = max(1, int(self.curriculum_steps))
        return float(np.clip(self.t_global / float(denom), 0.0, 1.0))

    def _update_effective_budget(self) -> None:
        p = self._progress_frac()
        frac = (1.0 - p) ** float(self.k_decay_power)
        k_eff = int(round(self.k_min_late + (self.K - self.k_min_late) * frac))
        self.K_eff = int(np.clip(k_eff, self.k_min_late, self.K))

    def _annealed_delta_scales(self) -> Dict[str, float]:
        p = self._progress_frac()
        frac = self.delta_min_frac + (1.0 - self.delta_min_frac) * ((1.0 - p) ** self.delta_anneal_power)
        return {
            "battery": float(self.delta_battery_scale0 * frac),
            "step_cost": float(self.delta_step_cost_scale0 * frac),
            "pslip": float(self.delta_pslip_scale0 * frac),
        }

    def _annealed_theta_reg_weight(self) -> float:
        p = self._progress_frac()
        w = self.theta_reg * (self.theta_reg_final_frac + (1.0 - self.theta_reg_final_frac) * (1.0 - p))
        return float(w)

    def _apply_curriculum(self) -> None:
        p = self._progress_frac()
        idx = 0
        for i, st in enumerate(self.curriculum_stages):
            if p <= float(st.get("until_frac", 1.0)):
                idx = i
                break
        st = self.curriculum_stages[idx]
        self._curr_stage_idx = int(idx)
        self._curr_stage_name = str(st.get("name", f"stage{idx}"))
        self._curr_slip_enabled = bool(st.get("slip_enabled", True))
        self._curr_allow_pslip_update = bool(st.get("allow_pslip_update", True))
        self._curr_query_every = int(st.get("query_every", self.query_every_base))

        if not self._curr_slip_enabled:
            self.theta.p_slip = 0.0
            clamp_theta(self.theta)

        self._update_effective_budget()

    # ---------------- obs / masking ----------------
    def _query_allowed(self) -> bool:
        return (self.since_query >= int(self._curr_query_every)) and (self.k_used_ep < int(self.K_eff))

    def _query_mask_scalar(self) -> float:
        return 1.0 if self._query_allowed() else 0.0

    def _sanitize_obs(self, vec: np.ndarray, where: str) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        if np.all(np.isfinite(vec)):
            return vec
        bad = np.where(~np.isfinite(vec))[0].tolist()
        rec = {
            "event": "warning",
            "where": where,
            "t_global": int(self.t_global),
            "ep": int(self.ep),
            "t_ep": int(self.t_ep),
            "bad_idx": bad,
            "theta_est": theta_to_dict(self.theta),
            "last_signals": [float(x) for x in np.nan_to_num(self.last_signals).tolist()],
        }
        self._log_f.write(json_dumps_safe(rec) + "\n")
        self._log_f.flush()
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _make_twin(self, seed: int) -> Any:
        cfg = RCConfig(
            seed=seed,
            grid_size=int(self.theta.grid_size),
            battery_max=int(self.theta.battery_max),
            step_cost=float(self.theta.step_cost),
            delivery_reward=float(self.theta.delivery_reward),
            battery_fail_penalty=float(self.theta.battery_fail_penalty),
            use_stay=bool(self.theta.use_stay),
        )
        base = make_env(cfg)
        if not self._curr_slip_enabled:
            return base
        try:
            return SlipActionWrapper(base, p_slip=float(self.theta.p_slip), seed=seed)
        except Exception:
            return base

    def _obs_vec(self) -> np.ndarray:
        th = self.theta
        theta_vec = np.array(
            [
                float(th.battery_max) / NORM_BATTERY_MAX,
                float(th.step_cost) / NORM_STEP_COST,
                float(th.p_slip) / NORM_P_SLIP,
            ],
            dtype=np.float32,
        )

        budget_frac = np.array([float(self.K_eff - self.k_used_ep) / max(1.0, float(self.K_eff))], dtype=np.float32)
        since_q = np.array([min(1.0, float(self.since_query) / 200.0)], dtype=np.float32)
        qmask = np.array([float(self._query_mask_scalar())], dtype=np.float32)
        stage = np.array(
            [float(self._curr_stage_idx) / max(1.0, float(len(self.curriculum_stages) - 1))],
            dtype=np.float32,
        )

        sig = self.last_signals
        if self.normalize_signals:
            # normalize signals online (helps PPO a lot)
            sig = self._signals_rms.normalize(sig, clip=self.signals_clip)

        obs = np.asarray(self.obs_t, dtype=np.float32)
        vec = np.concatenate([obs, theta_vec, budget_frac, since_q, sig, qmask, stage], axis=0)
        return vec

    def _driver_action(self, obs_t: np.ndarray) -> int:
        if self.driver_is_heuristic:
            return int(a5_driver_action(obs_t, self.grid_size, self.act_map, self.charge_thresh))
        a, _ = self.driver_model.predict(obs_t, deterministic=True)
        return int(a)

    # ---------------- penalties (NO GT) ----------------
    def _theta_reg_penalty(self) -> float:
        w = self._annealed_theta_reg_weight()
        th = self.theta
        pr = self.theta_prior

        db = ((float(th.battery_max) - float(pr.battery_max)) / NORM_BATTERY_MAX) ** 2
        ds = ((float(th.step_cost) - float(pr.step_cost)) / NORM_STEP_COST) ** 2
        dp = ((float(th.p_slip) - float(pr.p_slip)) / NORM_P_SLIP) ** 2

        return float(
            w
            * (
                self.theta_reg_battery_mult * db
                + self.theta_reg_step_mult * ds
                + self.theta_reg_pslip_mult * dp
            )
        )

    def _smoothness_penalties(self) -> Dict[str, float]:
        th = self.theta

        bat_pen = 0.0
        if self._prev_battery_for_smooth is None:
            self._prev_battery_for_smooth = float(th.battery_max)
        else:
            d = (float(th.battery_max) - float(self._prev_battery_for_smooth)) / NORM_BATTERY_MAX
            bat_pen = float(self.battery_smooth_w * (d * d))
            self._prev_battery_for_smooth = float(th.battery_max)

        sc_pen = 0.0
        if self._prev_step_cost_for_smooth is None:
            self._prev_step_cost_for_smooth = float(th.step_cost)
        else:
            d = (float(th.step_cost) - float(self._prev_step_cost_for_smooth)) / NORM_STEP_COST
            sc_pen = float(self.step_cost_smooth_w * (d * d))
            self._prev_step_cost_for_smooth = float(th.step_cost)

        ps_pen = 0.0
        if self._prev_pslip_for_smooth is None:
            self._prev_pslip_for_smooth = float(th.p_slip)
        else:
            d = (float(th.p_slip) - float(self._prev_pslip_for_smooth)) / NORM_P_SLIP
            ps_pen = float(self.pslip_smooth_w * (d * d))
            self._prev_pslip_for_smooth = float(th.p_slip)

        return {"battery_smooth": bat_pen, "step_cost_smooth": sc_pen, "pslip_smooth": ps_pen}

    def _bounds_barrier_penalty(self) -> float:
        th = self.theta
        eps = float(self.bounds_barrier_eps)

        def hinge_margin(xn: float) -> float:
            return max(0.0, eps - xn) ** 2 + max(0.0, xn - (1.0 - eps)) ** 2

        bxn = float(np.clip(float(th.battery_max) / NORM_BATTERY_MAX, 0.0, 1.0))
        sxn = float(np.clip(float(th.step_cost) / NORM_STEP_COST, 0.0, 1.0))
        pxn = float(np.clip(float(th.p_slip) / NORM_P_SLIP, 0.0, 1.0))

        pen = hinge_margin(bxn) + hinge_margin(sxn) + hinge_margin(pxn)
        return float(self.bounds_barrier_w * pen)

    # ---------------- query cost schedule ----------------
    def _adaptive_query_cost(self, acc_parts_no_gt: Dict[str, float], loss_parts: Dict[str, float]) -> float:
        pos_mse = float(acc_parts_no_gt.get("acc/pos_mse", 0.0) or 0.0)
        bat_mse = float(acc_parts_no_gt.get("acc/bat_mse", 0.0) or 0.0)
        rew_mae = float(acc_parts_no_gt.get("acc/rew_mae", 0.0) or 0.0)
        loss_total = float(loss_parts.get("loss/total", 0.0) or 0.0)

        bad = 0.0
        bad += 0.08 * pos_mse
        bad += 0.12 * bat_mse
        bad += 0.15 * rew_mae
        bad += 0.25 * loss_total

        p = self._progress_frac()
        if self.query_time_shaping:
            late_mult = float(math.exp(self.query_time_exp_k * p))
        else:
            # softer, mostly data-driven
            late_mult = 1.0 + 2.0 * p

        rem = float(self.K_eff - self.k_used_ep) / max(1.0, float(self.K_eff))
        budget_mult = 1.0 + 3.0 * ((1.0 - rem) ** 2)

        base = float(self.query_cost_base) * late_mult * budget_mult
        disc = 1.0 / (1.0 + 1.5 * bad)
        cost = base * disc

        min_cost = 0.01 * float(self.query_cost_base)
        return float(np.clip(cost, min_cost, base))

    # ---------------- gym API ----------------
    def reset(self, seed: int | None = None, options=None):
        if seed is None:
            seed = int(self._rng.integers(0, 10**7))

        self.ep += 1
        self.t_ep = 0
        self.k_used_ep = 0
        self.since_query = 10**9
        self.update_idx = 0
        self.prev_loss_total = None

        self._prev_battery_for_smooth = None
        self._prev_step_cost_for_smooth = None
        self._prev_pslip_for_smooth = None

        self.pair_buf = RingBuffer(capacity=max(256, self.batch_len * 4))
        self.last_signals = np.zeros((8,), dtype=np.float32)
        self._signals_rms = RunningMeanStd(shape=(8,))  # reset normalization per-run/episode for stability

        self._apply_curriculum()

        r = self.oracle.reset(seed=seed, config_overrides=None)
        self.sid = r["session_id"]

        ws = r.get("world_state", {}) or {}
        self.grid_size = int(ws.get("grid_size", self.theta.grid_size))
        self.theta.grid_size = self.grid_size
        self.theta.use_stay = bool(ws.get("use_stay", self.theta.use_stay))
        clamp_theta(self.theta)

        # GT subset (EVAL ONLY)
        bm = ws.get("battery_max", None)
        sc = ws.get("step_cost", None)
        ps = ws.get("p_slip", None)

        self.theta_true = {}
        if _is_valid_gt(bm):
            self.theta_true["battery_max"] = float(bm)
        if _is_valid_gt(sc):
            self.theta_true["step_cost"] = float(sc)
        self.gt_has_pslip = _is_valid_gt(ps)
        if self.gt_has_pslip:
            self.theta_true["p_slip"] = float(ps)

        # curriculum enforcement
        if not self._curr_slip_enabled:
            self.theta.p_slip = 0.0
            clamp_theta(self.theta)

        # twin reset
        self.twin = self._make_twin(seed=seed)
        obs0, _ = self.twin.reset(seed=seed)
        self.obs_t = np.asarray(obs0, dtype=np.float32)

        self._log_f.write(json_dumps_safe({
            "event": "reset",
            "run_id": getattr(self, "run_id", None),
            "ep": int(self.ep),
            "seed": int(seed),
            "theta_true_eval_only": dict(self.theta_true),
            "theta_est": theta_to_dict(self.theta),
            "K_eff": int(self.K_eff),
            "curriculum": {
                "stage_idx": int(self._curr_stage_idx),
                "stage_name": self._curr_stage_name,
                "slip_enabled": bool(self._curr_slip_enabled),
                "allow_pslip_update": bool(self._curr_allow_pslip_update),
                "query_every": int(self._curr_query_every),
            },
        }) + "\n")
        self._log_f.flush()

        obs_vec = self._sanitize_obs(self._obs_vec(), where="reset/obs_vec")
        return obs_vec, {}

    def step(self, action: np.ndarray):
        self._apply_curriculum()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        q_raw = float(action[0])

        if not self._query_allowed():
            q_raw = -1.0
        do_query = int(q_raw > 0.0)

        a_pslip = float(action[3])
        if not self._curr_allow_pslip_update:
            a_pslip = 0.0

        obs_prev = np.asarray(self.obs_t, dtype=np.float32)
        a_move = self._driver_action(obs_prev)

        obs_t2, r_t, term, trunc, _info_t = self.twin.step(int(a_move))
        obs_t2 = np.asarray(obs_t2, dtype=np.float32)
        self.obs_t = obs_t2

        reward = -float(self.step_penalty)

        if not do_query:
            self.since_query += 1
            self.t_ep += 1
            self.t_global += 1
            done = bool(term or trunc or (self.t_ep >= self.H))
            obs_vec = self._sanitize_obs(self._obs_vec(), where="step/no_query_obs_vec")
            return obs_vec, float(reward), done, False, {}

        # query path
        resp = self.oracle.step(self.sid, int(a_move))
        obs_o = np.asarray(resp.get("obs", resp.get("observation", [])), dtype=np.float32)
        r_o = float(resp.get("reward", 0.0))

        oracle_done = int(bool(resp.get("done", False) or resp.get("terminated", False) or resp.get("truncated", False)))
        twin_done = int(bool(term or trunc))

        twin_pos_next = np.array([float(obs_t2[0]), float(obs_t2[1])], dtype=float)
        oracle_pos_next = np.array([float(obs_o[0]), float(obs_o[1])], dtype=float)
        twin_bat_next = float(obs_t2[7])
        oracle_bat_next = float(obs_o[7])

        self.pair_buf.append(PairTransition(
            twin_pos_next=twin_pos_next,
            oracle_pos_next=oracle_pos_next,
            twin_bat_next=twin_bat_next,
            oracle_bat_next=oracle_bat_next,
            twin_rew=float(r_t),
            oracle_rew=float(r_o),
            twin_done=twin_done,
            oracle_done=oracle_done,
        ))

        recent = self.pair_buf.tail(self.batch_len)
        T = len(recent)

        batch = CalibBatch(
            pos_next_oracle=np.stack([x.oracle_pos_next for x in recent], axis=0) if T else np.zeros((0, 2)),
            pos_next_twin=np.stack([x.twin_pos_next for x in recent], axis=0) if T else np.zeros((0, 2)),
            bat_next_oracle=np.array([x.oracle_bat_next for x in recent], dtype=float) if T else np.zeros((0,)),
            bat_next_twin=np.array([x.twin_bat_next for x in recent], dtype=float) if T else np.zeros((0,)),
            reward_oracle=np.array([x.oracle_rew for x in recent], dtype=float) if T else np.zeros((0,)),
            reward_twin=np.array([x.twin_rew for x in recent], dtype=float) if T else np.zeros((0,)),
            done_oracle=np.array([x.oracle_done for x in recent], dtype=int) if T else np.zeros((0,), dtype=int),
            done_twin=np.array([x.twin_done for x in recent], dtype=int) if T else np.zeros((0,), dtype=int),
            slip_flag=None,
        )

        theta_est = theta_to_dict(self.theta)

        # Loss is GT-free by design.
        loss_parts = compute_loss_components(batch, theta_est, self.loss_weights, huber_delta=self.huber_delta)
        base_total = float(loss_parts.get("loss/total", 0.0) or 0.0)

        # CRITICAL: compute policy-facing acc WITHOUT GT (empty dict).
        acc_no_gt = compute_accuracy_metrics(batch, theta_est, {})  # <-- no GT leakage

        # Optional: compute eval acc with GT for logging only.
        acc_eval = compute_accuracy_metrics(batch, theta_est, dict(self.theta_true))

        # update last_signals for policy (NO GT-derived theta errors)
        new_signals = np.array([
            float(acc_no_gt.get("acc/pos_mse", 0.0) or 0.0),
            float(acc_no_gt.get("acc/bat_mse", 0.0) or 0.0),
            float(acc_no_gt.get("acc/rew_mae", 0.0) or 0.0),
            float(loss_parts.get("loss/pos", 0.0) or 0.0),
            float(loss_parts.get("loss/bat", 0.0) or 0.0),
            float(loss_parts.get("loss/rew", 0.0) or 0.0),
            float(loss_parts.get("loss/term", 0.0) or 0.0),
            float(base_total),
        ], dtype=np.float32)

        # update RMS before using in next obs
        if self.normalize_signals:
            self._signals_rms.update(new_signals)

        self.last_signals = new_signals

        qcost = float(self._adaptive_query_cost(acc_no_gt, loss_parts))

        if T < int(self.min_T):
            reg_pen = float(self._theta_reg_penalty())
            bnd_pen = float(self._bounds_barrier_penalty())
            reward = float(-qcost - reg_pen - bnd_pen - self.step_penalty)

            self.k_used_ep += 1
            self.k_used_total += 1
            self.since_query = 0
            self.update_idx += 1

            self._log_f.write(json_dumps_safe({
                "event": "oracle_update_skipped",
                "run_id": getattr(self, "run_id", None),
                "t_global": int(self.t_global),
                "update_idx": int(self.update_idx),
                "ep": int(self.ep),
                "t_ep": int(self.t_ep),
                "k_used_ep": int(self.k_used_ep),
                "k_used_total": int(self.k_used_total),
                "K_eff": int(self.K_eff),
                "batch_T": int(T),
                "reason": "insufficient_batch",
                "theta_true_eval_only": dict(self.theta_true),
                "theta_est": theta_to_dict(self.theta),
                "loss": loss_parts,
                "acc_no_gt": acc_no_gt,
                "acc_eval": acc_eval,
                "query_cost": float(qcost),
                "theta_reg": float(reg_pen),
                "bounds_barrier": float(bnd_pen),
                "reward": float(reward),
                "curriculum_stage": self._curr_stage_name,
                "query_mask": float(self._query_mask_scalar()),
            }) + "\n")
            self._log_f.flush()

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

            self.t_ep += 1
            self.t_global += 1
            done = bool(term or trunc or (self.t_ep >= self.H))
            obs_vec = self._sanitize_obs(self._obs_vec(), where="step/gate_insufficient_batch")
            return obs_vec, float(reward), done, False, {}

        # loss explode gate
        skip_theta_update = False
        if self.prev_loss_total is not None and base_total > 1.15 * float(self.prev_loss_total):
            skip_theta_update = True
        self.prev_loss_total = float(base_total)

        if not skip_theta_update:
            scales = self._annealed_delta_scales()
            a_b = float(np.tanh(float(action[1])))
            a_s = float(np.tanh(float(action[2])))
            a_p = float(np.tanh(float(a_pslip)))

            th = self.theta
            th.battery_max = float(th.battery_max) + float(scales["battery"]) * a_b
            th.step_cost = float(th.step_cost) + float(scales["step_cost"]) * a_s
            th.p_slip = float(th.p_slip) + float(scales["pslip"]) * a_p

            th.delivery_reward = float(self.theta_prior.delivery_reward)
            th.battery_fail_penalty = float(self.theta_prior.battery_fail_penalty)

            clamp_theta(th)
            self.theta = th
            apply_theta_to_env(self.twin, self.theta)

        reg_pen = float(self._theta_reg_penalty())
        smooth = self._smoothness_penalties()
        bnd_pen = float(self._bounds_barrier_penalty())

        total_obj = float(
            base_total
            + reg_pen
            + bnd_pen
            + smooth["battery_smooth"]
            + smooth["step_cost_smooth"]
            + smooth["pslip_smooth"]
        )
        reward = float(-total_obj - qcost - self.step_penalty)

        self.k_used_ep += 1
        self.k_used_total += 1
        self.since_query = 0
        self.update_idx += 1

        self._log_f.write(json_dumps_safe({
            "event": "oracle_update",
            "run_id": getattr(self, "run_id", None),
            "t_global": int(self.t_global),
            "update_idx": int(self.update_idx),
            "ep": int(self.ep),
            "t_ep": int(self.t_ep),
            "k_used_ep": int(self.k_used_ep),
            "k_used_total": int(self.k_used_total),
            "K_eff": int(self.K_eff),
            "batch_T": int(T),
            "theta_true_eval_only": dict(self.theta_true),
            "theta_est": theta_to_dict(self.theta),
            "loss": loss_parts,
            "acc_no_gt": acc_no_gt,
            "acc_eval": acc_eval,
            "query_cost": float(qcost),
            "theta_reg": float(reg_pen),
            "smooth": smooth,
            "bounds_barrier": float(bnd_pen),
            "base_loss_total": float(base_total),
            "total_obj": float(total_obj),
            "reward": float(reward),
            "curriculum_stage": self._curr_stage_name,
            "query_mask": float(self._query_mask_scalar()),
        }) + "\n")
        self._log_f.flush()

        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

        self.t_ep += 1
        self.t_global += 1
        done = bool(term or trunc or (self.t_ep >= self.H))
        obs_vec = self._sanitize_obs(self._obs_vec(), where="step/obs_vec")
        return obs_vec, float(reward), done, False, {}

    def close(self):
        try:
            self._log_f.close()
        except Exception:
            pass
        super().close()


def burn_in_rollout(env: ControllerEnv, steps: int, seed: int = 0) -> None:
    """
    Burn-in (NO PPO learning):
      - Query aggressively to fill buffer + stabilize signals
      - Tiny deltas to avoid poisoning.
    """
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    for _ in range(int(steps)):
        q_raw = 1.0
        a1 = float(np.clip(rng.normal(0.0, 0.05), -1.0, 1.0))
        a2 = float(np.clip(rng.normal(0.0, 0.05), -1.0, 1.0))
        a3 = float(np.clip(rng.normal(0.0, 0.05), -1.0, 1.0))

        act = np.array([q_raw, a1, a2, a3], dtype=np.float32)
        obs, r, done, _, _ = env.step(act)
        if done:
            obs, _ = env.reset(seed=int(rng.integers(0, 10**7)))


def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H = 1000
    K = 200
    TOTAL_STEPS = 50_000
    BURN_IN_STEPS = 7_500

    run_id = uuid.uuid4().hex[:8]
    log_path = f"logs/ppo_calib_{run_id}.jsonl"

    DRIVER_MODEL = "models/ppo_driver_5fffd98de8.zip"

    init_theta = Theta(battery_max=80, step_cost=0.10, delivery_reward=10.0, battery_fail_penalty=8.0, p_slip=0.10)
    prior_theta = Theta(battery_max=80, step_cost=0.10, delivery_reward=10.0, battery_fail_penalty=8.0, p_slip=0.10)

    env = ControllerEnv(
        oracle_url=ORACLE_URL,
        driver_model_path=DRIVER_MODEL,
        log_path=log_path,
        H=H,
        K=K,
        query_every=20,
        sleep_s=0.0,
        query_cost_base=0.10,
        step_penalty=0.001,
        theta_reg=0.25,
        theta_reg_final_frac=0.10,
        theta_reg_battery_mult=0.25,
        delta_step_cost_scale=0.010,
        delta_min_frac=0.20,
        delta_anneal_power=1.0,
        battery_smooth_w=0.30,
        step_cost_smooth_w=0.50,
        pslip_smooth_w=0.20,
        bounds_barrier_w=0.25,
        bounds_barrier_eps=0.08,
        k_min_late=10,
        k_decay_power=2.0,
        normalize_signals=True,     # важлива стабілізація
        signals_clip=5.0,
        query_time_shaping=True,    # можна вимкнути для "більш RL, менше schedule"
        query_time_exp_k=3.0,
        init_theta=init_theta,
        prior_theta=prior_theta,
        curriculum_steps=TOTAL_STEPS,
    )
    env.run_id = run_id

    print(f"Run ID: {run_id}")
    print(f"Logging: {log_path}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write('{"event":"test_connection"}\n')
    print("Test record written.")

    print(f"Burn-in rollout: {BURN_IN_STEPS} steps (no PPO learning)")
    burn_in_rollout(env, steps=BURN_IN_STEPS, seed=env.seed0)
    print("Burn-in done.")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    )

    print(f"Training started. Total timesteps: {TOTAL_STEPS}")
    model.learn(total_timesteps=TOTAL_STEPS)

    model_file = f"models/ppo_controller_{run_id}.zip"
    model.save(model_file)
    env.close()

    print(f"\nDone. Model: {model_file}")
    print(f"Now plot: {log_path}")
    print("Notes:")
    print(" - GT leakage removed: policy acc is computed with empty theta_true (acc_no_gt).")
    print(" - acc_eval uses theta_true only for logging/evaluation.")
    print(" - last_signals are online-normalized for stable PPO learning.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from env import RCConfig, make_env

from rc_calib.wrappers import SlipActionWrapper

from metrics import CalibBatch, compute_loss_components, compute_accuracy_metrics

from rl_utils import (
    RoboCourierOracle,
    Theta, clamp_theta, theta_to_dict,
    infer_action_mapping_local,
    obs_to_robot_xy, norm_xy_to_grid, greedy_action_towards,
    apply_theta_to_env,
)


# Multi-loss weights (slip intentionally disabled by default)
LOSS_WEIGHTS = {"pos": 1.0, "bat": 2.0, "rew": 1.0, "term": 1.0, "slip": 0.0}
CALIB_BATCH_LEN = 32
HUBER_DELTA = 1.0

# Logging
LOG_UPDATE_ONLY = True
LOG_STEPS = False  # True only for debugging


# Utilities

def a5_driver_action(obs: np.ndarray, grid_size: int, act_map: Dict[str, int], charge_thresh: float = 0.30) -> int:
    """A5-like heuristic driver used to generate actions in the environment."""
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


# Ring buffer for paired oracle transitions
@dataclass
class PairTransition:
    twin_pos_next: np.ndarray   # [2]
    oracle_pos_next: np.ndarray # [2]
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


# ControllerEnv
class ControllerEnv(gym.Env):
    """
    PPO controller chooses:
      action[0] = query gate (q_raw)
      action[1] = delta for battery_max
      action[2] = delta for step_cost
      action[3] = delta for p_slip

    Only applies theta updates on oracle queries.
    Computes batch loss over last CALIB_BATCH_LEN oracle transitions.
    Logs update-only records for clean plots.
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
        theta_reg: float = 0.15,
        charge_thresh: float = 0.30,
        init_theta: Optional[Theta] = None,
        prior_theta: Optional[Theta] = None,
    ):
        super().__init__()
        self.prev_loss_total: Optional[float] = None
        self.oracle = RoboCourierOracle(oracle_url)
        self.seed0 = int(seed0)

        self.H = int(H)
        self.K = int(K)
        self.query_every = int(query_every)
        self.sleep_s = float(sleep_s)

        self.query_cost_base = float(query_cost_base)
        self.step_penalty = float(step_penalty)
        self.theta_reg = float(theta_reg)
        self.charge_thresh = float(charge_thresh)

        # Multi-loss config
        self.loss_weights = dict(LOSS_WEIGHTS)
        self.huber_delta = float(HUBER_DELTA)
        self.batch_len = int(CALIB_BATCH_LEN)
        self.min_T = max(8, self.batch_len // 2)  # soft gate for early updates


        # Logging
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_f = self.log_path.open("w", encoding="utf-8")

        # Action mapping
        self.act_map = infer_action_mapping_local(seed=self.seed0)

        # Driver model
        self.driver_is_heuristic = (driver_model_path is None)
        self.driver_model = PPO.load(driver_model_path) if (not self.driver_is_heuristic) else None

        # Controller action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # twin_obs(10) + theta(3) + budget_frac(1) + since_query(1) + last_acc(4)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(10 + 3 + 1 + 1 + 4,), dtype=np.float32
        )

        self.theta = init_theta if init_theta is not None else Theta()
        self.theta_prior = prior_theta if prior_theta is not None else Theta(
            battery_max=80, step_cost=0.10, delivery_reward=10.0, battery_fail_penalty=8.0, p_slip=0.10
        )

        self._rng = np.random.default_rng(self.seed0)

        # Runtime state
        self.sid: Optional[str] = None
        self.grid_size: int = 10

        self.ep: int = 0
        self.t_ep: int = 0
        self.t_global: int = 0

        self.k_used_ep: int = 0
        self.k_used_total: int = 0
        self.since_query: int = 10**9

        #never store NaN here
        self.last_acc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.obs_t: Optional[np.ndarray] = None
        self.twin = None

        # Oracle GT theta (subset)
        self.theta_true: Dict[str, Any] = {}
        self.gt_has_pslip: bool = False

        # Buffer of paired oracle transitions
        self.pair_buf = RingBuffer(capacity=max(256, self.batch_len * 4))

        # Update index for queries
        self.update_idx: int = 0

    # Safe observation
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
            "bad_values": [None if not np.isfinite(float(vec[i])) else float(vec[i]) for i in bad],
            "theta_est": theta_to_dict(self.theta),
            "last_acc": [float(x) for x in np.nan_to_num(self.last_acc, nan=0.0, posinf=0.0, neginf=0.0).tolist()],
        }
        self._log_f.write(json_dumps_safe(rec) + "\n")
        self._log_f.flush()

        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Twin env builder
    def _make_twin(self, seed: int) -> Any:
        cfg = RCConfig(
            seed=seed,
            grid_size=int(self.theta.grid_size),
            battery_max=int(self.theta.battery_max),
            step_cost=float(self.theta.step_cost),
            delivery_reward=float(self.theta.delivery_reward),            # fixed
            battery_fail_penalty=float(self.theta.battery_fail_penalty),  # fixed
            use_stay=bool(self.theta.use_stay),
        )
        base = make_env(cfg)

        # Optional slip wrapper
        try:
            twin = SlipActionWrapper(base, p_slip=float(self.theta.p_slip), seed=seed)
            return twin
        except Exception:
            return base

    def _obs_vec(self) -> np.ndarray:
        th = self.theta

        # normalize theta components
        theta_vec = np.array(
            [
                float(th.battery_max) / 400.0,
                float(th.step_cost) / 0.50,
                float(th.p_slip) / 0.30,
            ],
            dtype=np.float32,
        )

        budget_frac = np.array(
            [float(self.K - self.k_used_ep) / max(1.0, float(self.K))],
            dtype=np.float32,
        )
        since_q = np.array([min(1.0, float(self.since_query) / 200.0)], dtype=np.float32)

        obs = np.asarray(self.obs_t, dtype=np.float32)
        vec = np.concatenate([obs, theta_vec, budget_frac, since_q, self.last_acc], axis=0)
        return vec

    def _driver_action(self, obs_t: np.ndarray) -> int:
        if self.driver_is_heuristic:
            return int(a5_driver_action(obs_t, self.grid_size, self.act_map, self.charge_thresh))
        a, _ = self.driver_model.predict(obs_t, deterministic=True)
        return int(a)

    def _theta_reg_penalty(self) -> float:
        th = self.theta
        pr = self.theta_prior
        db = ((float(th.battery_max) - float(pr.battery_max)) / 400.0) ** 2
        ds = ((float(th.step_cost) - float(pr.step_cost)) / 0.50) ** 2
        dp = ((float(th.p_slip) - float(pr.p_slip)) / 0.30) ** 2
        return float(self.theta_reg * (db + ds + dp))

    def _adaptive_query_cost(self, acc_parts: Dict[str, float]) -> float:
        theta_ma = acc_parts.get("acc/theta_mean_abs", 0.0)
        pos_mse = acc_parts.get("acc/pos_mse", 0.0)
        bat_mse = acc_parts.get("acc/bat_mse", 0.0)
        rew_mae = acc_parts.get("acc/rew_mae", 0.0)

        # sanitize NaNs
        if isinstance(theta_ma, float) and math.isnan(theta_ma): theta_ma = 0.0
        if isinstance(pos_mse, float) and math.isnan(pos_mse): pos_mse = 0.0
        if isinstance(bat_mse, float) and math.isnan(bat_mse): bat_mse = 0.0
        if isinstance(rew_mae, float) and math.isnan(rew_mae): rew_mae = 0.0

        bad = 0.0
        bad += float(theta_ma)
        bad += 0.05 * float(pos_mse)
        bad += 0.05 * float(bat_mse)
        bad += 0.10 * float(rew_mae)

        # Keep >0 for stability
        min_cost = 0.01 * float(self.query_cost_base)
        cost = float(self.query_cost_base) / (1.0 + bad)
        return float(np.clip(cost, min_cost, float(self.query_cost_base)))

    # Gym API
    def reset(self, seed: int | None = None, options=None):
        if seed is None:
            seed = int(self._rng.integers(0, 10**7))

        self.ep += 1
        self.t_ep = 0
        self.k_used_ep = 0
        self.since_query = 10**9
        self.last_acc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.pair_buf = RingBuffer(capacity=max(256, self.batch_len * 4))
        self.update_idx = 0

        r = self.oracle.reset(seed=seed, config_overrides=None)
        self.sid = r["session_id"]

        ws = r.get("world_state", {}) or {}
        self.grid_size = int(ws.get("grid_size", self.theta.grid_size))
        self.theta.grid_size = self.grid_size
        self.theta.use_stay = bool(ws.get("use_stay", self.theta.use_stay))
        clamp_theta(self.theta)

        # Extract GT subset 
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

        # Twin reset
        self.twin = self._make_twin(seed=seed)
        obs0, _ = self.twin.reset(seed=seed)
        self.obs_t = np.asarray(obs0, dtype=np.float32)

        # Log reset
        self._log_f.write(
            json_dumps_safe(
                {
                    "event": "reset",
                    "run_id": getattr(self, "run_id", None),
                    "ep": self.ep,
                    "seed": int(seed),
                    "theta_true": dict(self.theta_true),
                    "theta_est": theta_to_dict(self.theta),
                    "gt_has_pslip": bool(self.gt_has_pslip),
                }
            )
            + "\n"
        )
        self._log_f.flush()

        obs_vec = self._sanitize_obs(self._obs_vec(), where="reset/obs_vec")
        return obs_vec, {}



    def step(self, action: np.ndarray):
        import time  # safe even if already imported elsewhere

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        q_raw = float(action[0])

        query_allowed = (self.since_query >= self.query_every) and (self.k_used_ep < self.K)
        do_query = int((q_raw > 0.0) and query_allowed)

        obs_prev = np.asarray(self.obs_t, dtype=np.float32)
        a_move = self._driver_action(obs_prev)

        obs_t2, r_t, term, trunc, _info_t = self.twin.step(int(a_move))
        obs_t2 = np.asarray(obs_t2, dtype=np.float32)
        self.obs_t = obs_t2

        # base step penalty always
        reward = -float(self.step_penalty)

        #NO QUERY PATH
        if not do_query:
            self.since_query += 1
            self.t_ep += 1
            self.t_global += 1

            done = bool(term or trunc or (self.t_ep >= self.H))

            if (not LOG_UPDATE_ONLY) and LOG_STEPS:
                self._log_f.write(
                    json_dumps_safe({
                        "event": "step",
                        "t_global": int(self.t_global),
                        "ep": int(self.ep),
                        "t_ep": int(self.t_ep),
                        "did_query": 0,
                        "k_used_ep": int(self.k_used_ep),
                        "k_used_total": int(self.k_used_total),
                        "since_query": int(self.since_query),
                        "theta_est": theta_to_dict(self.theta),
                        "reward": float(reward),
                    }) + "\n"
                )
                self._log_f.flush()

            obs_vec = self._sanitize_obs(self._obs_vec(), where="step/no_query_obs_vec")
            return obs_vec, float(reward), done, False, {}

        #QUERY PATH STEP ORACLE
        resp = self.oracle.step(self.sid, int(a_move))

        obs_o = np.asarray(resp.get("obs", resp.get("observation", [])), dtype=np.float32)
        r_o = float(resp.get("reward", 0.0))

        oracle_done = int(bool(resp.get("done", False) or resp.get("terminated", False) or resp.get("truncated", False)))
        twin_done = int(bool(term or trunc))

        # extract next states
        twin_pos_next = np.array([float(obs_t2[0]), float(obs_t2[1])], dtype=float)
        oracle_pos_next = np.array([float(obs_o[0]), float(obs_o[1])], dtype=float)
        twin_bat_next = float(obs_t2[7])
        oracle_bat_next = float(obs_o[7])

        #PairTransition must have these fields
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

        #Build calibration batch
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
        theta_true = dict(self.theta_true)

        loss_parts = compute_loss_components(batch, theta_est, self.loss_weights, huber_delta=self.huber_delta)

        # ===== GATE: loss not improving (soft) =====
        skip_theta_update = False
        if self.prev_loss_total is not None:
            if float(loss_parts.get("loss/total", 0.0)) > 1.2 * float(self.prev_loss_total):
                skip_theta_update = True
        self.prev_loss_total = float(loss_parts.get("loss/total", 0.0))

        acc_parts = compute_accuracy_metrics(batch, theta_est, theta_true)

        # update last_acc (ffill)
        theta_ma = acc_parts.get("acc/theta_mean_abs", 0.0)
        if isinstance(theta_ma, float) and math.isnan(theta_ma):
            theta_ma = 0.0

        self.last_acc = np.array([
            float(acc_parts.get("acc/pos_mse", 0.0) or 0.0),
            float(acc_parts.get("acc/bat_mse", 0.0) or 0.0),
            float(acc_parts.get("acc/rew_mae", 0.0) or 0.0),
            float(theta_ma),
        ], dtype=np.float32)

        # adaptive query cost based on current accuracy
        qcost = float(self._adaptive_query_cost(acc_parts))

        # SOFT GATE: allow updates after we have at least min_T samples
        #define self.min_T in init  or keep the local fallback below
        min_T = max(8, self.batch_len // 2)
        if T < min_T:
            reg_pen = float(self._theta_reg_penalty())
            reward = float(-qcost - reg_pen - self.step_penalty)

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
                "batch_T": int(T),
                "reason": "insufficient_batch",
                "theta_true": theta_true,
                "theta_est": theta_to_dict(self.theta),
                "loss": loss_parts,
                "acc": acc_parts,
                "query_cost": float(qcost),
                "theta_reg": float(reg_pen),
                "reward": float(reward),
            }) + "\n")
            self._log_f.flush()

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

            self.t_ep += 1
            self.t_global += 1
            done = bool(term or trunc or (self.t_ep >= self.H))
            obs_vec = self._sanitize_obs(self._obs_vec(), where="step/gate_insufficient_batch")
            return obs_vec, float(reward), done, False, {}

        #THETA UPDATE only when batch is ready and loss is not exploding
        th = self.theta
        if not skip_theta_update:
            th.battery_max = float(th.battery_max) + 15.0 * float(action[1])
            th.step_cost   = float(th.step_cost)   + 0.02 * float(action[2])
            th.p_slip      = float(th.p_slip)      + 0.01 * float(action[3])

        # keep reward params fixed
        th.delivery_reward = float(self.theta_prior.delivery_reward)
        th.battery_fail_penalty = float(self.theta_prior.battery_fail_penalty)

        clamp_theta(th)
        self.theta = th
        apply_theta_to_env(self.twin, self.theta)

        reg_pen = float(self._theta_reg_penalty())
        reward = float(-float(loss_parts.get("loss/total", 0.0)) - qcost - reg_pen - self.step_penalty)

        self.k_used_ep += 1
        self.k_used_total += 1
        self.since_query = 0
        self.update_idx += 1

        # log update
        log_event = {
            "event": "oracle_update",
            "run_id": getattr(self, "run_id", None),
            "t_global": int(self.t_global),
            "update_idx": int(self.update_idx),
            "ep": int(self.ep),
            "t_ep": int(self.t_ep),
            "k_used_ep": int(self.k_used_ep),
            "k_used_total": int(self.k_used_total),
            "batch_T": int(T),
            "theta_true": theta_true,
            "theta_est": theta_to_dict(self.theta),
            "loss": loss_parts,
            "acc": acc_parts,
            "query_cost": float(qcost),
            "theta_reg": float(reg_pen),
            "reward": float(reward),
        }
        self._log_f.write(json_dumps_safe(log_event) + "\n")

        if (not LOG_UPDATE_ONLY) and LOG_STEPS:
            self._log_f.write(json_dumps_safe({
                "event": "step",
                "t_global": int(self.t_global),
                "ep": int(self.ep),
                "t_ep": int(self.t_ep),
                "did_query": 1,
                "k_used_ep": int(self.k_used_ep),
                "k_used_total": int(self.k_used_total),
                "since_query": int(self.since_query),
                "theta_est": theta_to_dict(self.theta),
                "reward": float(reward),
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


def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H = 1000
    K = 200
    TOTAL_STEPS = 50_000

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
        theta_reg=0.15,
        init_theta=init_theta,
        prior_theta=prior_theta,
    )
    env.run_id = run_id

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print(f"Training started. Run ID: {run_id}")
    print(f"Logging: {log_path}")

    model.learn(total_timesteps=TOTAL_STEPS)

    model_file = f"models/ppo_controller_{run_id}.zip"
    model.save(model_file)
    env.close()

    print(f"\nDone. Model: {model_file}")
    print(f"Now run your plot script on: {log_path}")
    print("Note: updates are logged as event='oracle_update' with fields loss/* and acc/*")


if __name__ == "__main__":
    main()

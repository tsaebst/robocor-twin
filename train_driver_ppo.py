from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from env import RCConfig, make_env
from rc_calib.wrappers import SlipActionWrapper 

from rl_utils import (
    RoboCourierOracle,
    Theta, clamp_theta, theta_to_dict,
    infer_action_mapping_local,
    obs_to_robot_xy, norm_xy_to_grid, greedy_action_towards,
    apply_theta_to_env,
)

# Heuristic driver
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

# Safe JSON
def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj

def json_dumps_safe(obj: Any) -> str:
    return json.dumps(_sanitize(obj), ensure_ascii=False)

# PPO chooses quer+theta deltas
class ControllerEnv(gym.Env):
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

        # reward weights 
        w_pos: float = 1.0,
        w_bat: float = 1.0,
        w_rew: float = 0.1,

        # query cost
        query_cost_base: float = 0.05,
        query_cost_freq: float = 0.10,   # extra cost when querying too frequently

        # small step penalty
        step_penalty: float = 0.001,

        # driver heuristic param
        charge_thresh: float = 0.30,

        init_theta: Optional[Theta] = None,):
        super().__init__()
        self.oracle = RoboCourierOracle(oracle_url)
        self.seed0 = int(seed0)

        self.H = int(H)
        self.K = int(K)
        self.query_every = int(query_every)
        self.sleep_s = float(sleep_s)

        self.w_pos = float(w_pos)
        self.w_bat = float(w_bat)
        self.w_rew = float(w_rew)

        self.query_cost_base = float(query_cost_base)
        self.query_cost_freq = float(query_cost_freq)
        self.step_penalty = float(step_penalty)

        self.charge_thresh = float(charge_thresh)

        # logging
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_f = self.log_path.open("w", encoding="utf-8")

        # action mapping
        self.act_map = infer_action_mapping_local(seed=self.seed0)

        # driver model
        self.driver_is_heuristic = (driver_model_path is None)
        self.driver_model = PPO.load(driver_model_path) if (not self.driver_is_heuristic) else None

        # controller action q_raw, d_bmax, d_step_cost, d_deliv, d_fail, d_pslip
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # observation twin_obs(10) + theta(5) + budget_frac(1) + since_query(1) + last_err(3)
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(10 + 5 + 1 + 1 + 3,), dtype=np.float32)

        self.theta = init_theta if init_theta is not None else Theta()
        self._rng = np.random.default_rng(self.seed0)

        # runtime state
        self.sid: Optional[str] = None
        self.grid_size: int = 10

        self.ep: int = 0
        self.t_ep: int = 0
        self.t_global: int = 0

        self.k_used_ep: int = 0
        self.k_used_total: int = 0

        self.since_query: int = 10**9
        self.last_err = np.zeros(3, dtype=np.float32)

        self.obs_t: Optional[np.ndarray] = None
        self.twin = None

        # oracle GT theta subset
        self.theta_true: Dict[str, Any] = {}
        self.gt_has_pslip: bool = False  # auto-detected each episode

        # loss tracking for improvement reward
        self.prev_loss: Optional[float] = None

    # Build twin env
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
        try:
            twin = SlipActionWrapper(base, p_slip=float(self.theta.p_slip), seed=seed)
            return twin
        except Exception:
            return base

    def _obs_vec(self) -> np.ndarray:
        th = self.theta
        theta_vec = np.array(
            [
                float(th.battery_max) / 400.0,
                float(th.step_cost),
                float(th.delivery_reward) / 20.0,
                float(th.battery_fail_penalty) / 25.0,
                float(th.p_slip) / 0.30,
            ],
            dtype=np.float32,
        )
        budget_frac = np.array([float(self.K - self.k_used_ep) / max(1.0, float(self.K))], dtype=np.float32)
        since_q = np.array([min(1.0, float(self.since_query) / 200.0)], dtype=np.float32)

        obs = np.asarray(self.obs_t, dtype=np.float32)
        return np.concatenate([obs, theta_vec, budget_frac, since_q, self.last_err], axis=0)

    def _driver_action(self, obs_t: np.ndarray) -> int:
        if self.driver_is_heuristic:
            return int(a5_driver_action(obs_t, self.grid_size, self.act_map, self.charge_thresh))
        a, _ = self.driver_model.predict(obs_t, deterministic=True)
        return int(a)

    # Metrics theta vs GT with auto-excluding p_slip if GT missing
    def _theta_metrics(self) -> Dict[str, float]:
        eps = 1e-6
        # only parameters with valid GT
        keys: List[str] = ["battery_max", "step_cost"]
        if self.gt_has_pslip:
            keys.append("p_slip")

        tru = self.theta_true
        est = theta_to_dict(self.theta)

        #bounds for normalized L2 
        #must match clamp ranges
        bounds = {
            "battery_max": (30.0, 400.0),
            "step_cost": (0.01, 0.50),
            "p_slip": (0.0, 0.30),
        }

        rpes = []
        n_est = []
        n_tru = []
        for k in keys:
            tv = float(tru[k])
            ev = float(est[k])
            rpes.append(abs(ev - tv) / (abs(tv) + eps))
            lo, hi = bounds[k]
            n_est.append((ev - lo) / (hi - lo))
            n_tru.append((tv - lo) / (hi - lo))

        mean_rpe = float(np.mean(rpes)) if rpes else float("nan")
        n_l2 = float(np.linalg.norm(np.array(n_est) - np.array(n_tru))) if rpes else float("nan")

        return {
            "mean_rpe": mean_rpe,
            "n_l2": n_l2,
        }

    # Loss for improvement reward
    def _loss_from_err(self, pos_mse: float, bat_mse: float, rew_abs_err: float) -> float:
        return float(self.w_pos * pos_mse + self.w_bat * bat_mse + self.w_rew * rew_abs_err)

    def _query_cost(self) -> float:
        #  querying right after a query costs more
        # since_query < query_every -> extra cost close to query_cost_freq
        if self.since_query >= self.query_every:
            extra = 0.0
        else:
            extra = self.query_cost_freq * (1.0 - (self.since_query / max(1.0, float(self.query_every))))
        return float(self.query_cost_base + extra)

    # Gym API
    def reset(self, seed: int | None = None, options=None):
        if seed is None:
            seed = int(self._rng.integers(0, 10**7))

        self.ep += 1
        self.t_ep = 0
        self.k_used_ep = 0
        self.since_query = 10**9
        self.last_err = np.zeros(3, dtype=np.float32)
        self.prev_loss = None

        r = self.oracle.reset(seed=seed, config_overrides=None)
        self.sid = r["session_id"]

        ws = r.get("world_state", {}) or {}
        self.grid_size = int(ws.get("grid_size", self.theta.grid_size))
        self.theta.grid_size = self.grid_size
        self.theta.use_stay = bool(ws.get("use_stay", self.theta.use_stay))
        clamp_theta(self.theta)

        # extract theta_true; p_slip may be absent/NaN
        bm = ws.get("battery_max", None)
        sc = ws.get("step_cost", None)
        ps = ws.get("p_slip", None)

        def _is_valid(x):
            if x is None:
                return False
            if isinstance(x, float) and math.isnan(x):
                return False
            return True

        self.theta_true = {}
        if _is_valid(bm):
            self.theta_true["battery_max"] = float(bm)
        if _is_valid(sc):
            self.theta_true["step_cost"] = float(sc)

        self.gt_has_pslip = _is_valid(ps)
        if self.gt_has_pslip:
            self.theta_true["p_slip"] = float(ps)

        # twin
        self.twin = self._make_twin(seed=seed)
        obs0, _ = self.twin.reset(seed=seed)
        self.obs_t = np.asarray(obs0, dtype=np.float32)

        # log reset event
        self._log_f.write(
            json_dumps_safe(
                {
                    "event": "reset",
                    "run_id": getattr(self, "run_id", None),
                    "ep": self.ep,
                    "seed": int(seed),
                    "theta_true": self.theta_true | {"p_slip": (float(ps) if self.gt_has_pslip else None)},
                    "theta_est": theta_to_dict(self.theta),
                    "gt_has_pslip": bool(self.gt_has_pslip),
                }
            )
            + "\n"
        )
        self._log_f.flush()

        return self._obs_vec(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        q_raw = float(action[0])

        #controller may propose query, but we enforce min interval + budget
        query_allowed = (self.since_query >= self.query_every) and (self.k_used_ep < self.K)
        do_query = int((q_raw > 0.0) and query_allowed)

        # driver step
        a_move = self._driver_action(np.asarray(self.obs_t, dtype=np.float32))
        obs_t2, r_t, term, trunc, _ = self.twin.step(int(a_move))
        self.obs_t = np.asarray(obs_t2, dtype=np.float32)

        # default reward: time penalty
        reward = -float(self.step_penalty)

        #compute/update only on query
        update_rec = None
        if do_query:
            resp = self.oracle.step(self.sid, int(a_move))
            obs_o = np.asarray(resp["obs"], dtype=np.float32)
            r_o = float(resp["reward"])

            pos_mse = float(np.mean((obs_o[0:2] - self.obs_t[0:2]) ** 2))
            bat_mse = float((obs_o[7] - self.obs_t[7]) ** 2)
            rew_abs_err = float(abs(r_o - float(r_t)))

            self.last_err = np.array([pos_mse, bat_mse, rew_abs_err], dtype=np.float32)

            loss_new = self._loss_from_err(pos_mse, bat_mse, rew_abs_err)
            if self.prev_loss is None:
                improvement = 0.0
            else:
                improvement = float(self.prev_loss - loss_new)
            self.prev_loss = float(loss_new)

            qcost = self._query_cost()
            reward = float(improvement - qcost)

            # update theta ONLY on query
            # if GT for p_slip is NaN -> still may keep p_slip in theta dynamics
            # but do NOT include it in theta accuracy metrics
            th = self.theta
            th.battery_max = int(round(th.battery_max + 80.0 * float(action[1])))
            th.step_cost = float(th.step_cost + 0.10 * float(action[2]))
            th.delivery_reward = float(th.delivery_reward + 6.0 * float(action[3]))
            th.battery_fail_penalty = float(th.battery_fail_penalty + 8.0 * float(action[4]))
            th.p_slip = float(th.p_slip + 0.05 * float(action[5]))

            clamp_theta(th)
            self.theta = th
            apply_theta_to_env(self.twin, self.theta)

            # counters
            self.k_used_ep += 1
            self.k_used_total += 1
            self.since_query = 0

            # metrics
            theta_metrics = self._theta_metrics()
            update_rec = {
                "event": "update",
                "t_global": int(self.t_global),
                "ep": int(self.ep),
                "t_ep": int(self.t_ep),
                "did_query": 1,
                "k_used_ep": int(self.k_used_ep),
                "k_used_total": int(self.k_used_total),
                "theta_est": theta_to_dict(self.theta),
                "theta_true": self.theta_true,
                "gt_has_pslip": bool(self.gt_has_pslip),
                "state_err": {
                    "pos_mse": float(pos_mse),
                    "bat_mse": float(bat_mse),
                    "reward_abs_err": float(rew_abs_err),
                    "loss_new": float(loss_new),
                    "improvement": float(improvement),
                    "query_cost": float(qcost),
                },
                "theta_metrics": theta_metrics,
                "reward": float(reward),
            }

            if self.sleep_s > 0:
                import time
                time.sleep(self.sleep_s)
        else:
            self.since_query += 1

        # log step-level record EVERY step
        step_rec = {
            "event": "step",
            "t_global": int(self.t_global),
            "ep": int(self.ep),
            "t_ep": int(self.t_ep),
            "did_query": int(do_query),
            "k_used_ep": int(self.k_used_ep),
            "k_used_total": int(self.k_used_total),
            "since_query": int(self.since_query),
            "theta_est": theta_to_dict(self.theta),
            "last_err": [float(x) for x in self.last_err.tolist()],
            "reward": float(reward),
        }
        self._log_f.write(json_dumps_safe(step_rec) + "\n")
        if update_rec is not None:
            self._log_f.write(json_dumps_safe(update_rec) + "\n")
        self._log_f.flush()

        # advance time
        self.t_ep += 1
        self.t_global += 1

        done = bool(term or trunc or (self.t_ep >= self.H))
        return self._obs_vec(), float(reward), done, False, {}

    def close(self):
        try:
            self._log_f.close()
        except Exception:
            pass
        super().close()

def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    # fast debug params
    H = 1000
    K = 200
    TOTAL_STEPS = 50_000

    run_id = uuid.uuid4().hex[:8]
    log_path = f"logs/ppo_calib_{run_id}.jsonl"

    # driver model
    DRIVER_MODEL = "models/ppo_driver_5fffd98de8.zip" 

    env = ControllerEnv(
        oracle_url=ORACLE_URL,
        driver_model_path=DRIVER_MODEL,
        log_path=log_path,
        H=H,
        K=K,
        query_every=20,
        sleep_s=0.0,
        w_pos=1.0,
        w_bat=1.0,
        w_rew=0.1,
        query_cost_base=0.10, 
        query_cost_freq=0.30, 
        step_penalty=0.001,
        init_theta=Theta(battery_max=80, step_cost=0.10, delivery_reward=10.0, battery_fail_penalty=8.0, p_slip=0.10),
    )
    env.run_id = run_id  #for logging

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
    print(f"Now run: python plot_ppo_calib_report.py --log {log_path}")

if __name__ == "__main__":
    main()

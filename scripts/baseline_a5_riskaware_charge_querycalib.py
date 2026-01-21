from __future__ import annotations

import json
import time
import uuid
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from env import RCConfig, make_env
from wrappers import SlipActionWrapper


# =========================
# Oracle client (HTTP) with retries/backoff
# =========================
class RoboCourierOracle:
    def __init__(self, base_url: str, timeout_s: float = 30.0, max_retries: int = 6):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)

    def _post(self, path: str, payload: dict) -> dict:
        last_err: Optional[BaseException] = None
        for i in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    timeout=self.timeout_s,
                )
                r.raise_for_status()
                return r.json()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError,
            ) as e:
                last_err = e
                # exponential backoff with jitter
                sleep_s = min(8.0, 0.35 * (2 ** i)) * (0.7 + 0.6 * random.random())
                time.sleep(sleep_s)
        raise RuntimeError(f"Oracle request failed after {self.max_retries} retries: {path}") from last_err

    def reset(self, seed: int = 0, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"seed": int(seed), "config_overrides": config_overrides}
        return self._post("/reset", payload)

    def step(self, session_id: str, action: int) -> Dict[str, Any]:
        payload = {"session_id": str(session_id), "action": int(action)}
        return self._post("/step", payload)


# =========================
# Env unwrap + state set
# =========================
def unwrap_env(env_obj: Any, max_depth: int = 30) -> Any:
    base = env_obj
    for _ in range(max_depth):
        if hasattr(base, "env"):
            base = getattr(base, "env")
        else:
            break
    return base

def force_world_state_on_env(env_obj: Any, state: Dict[str, Any]) -> None:
    base = unwrap_env(env_obj)
    base.rx, base.ry = int(state["rx"]), int(state["ry"])
    base.px, base.py = int(state["px"]), int(state["py"])
    base.dx, base.dy = int(state["dx"]), int(state["dy"])
    base.cx, base.cy = int(state["cx"]), int(state["cy"])
    base.battery = int(state["battery"])
    base.has_package = bool(state["has_package"])


# =========================
# Coords + action mapping
# =========================
def norm_xy_to_grid(xn: float, yn: float, grid_size: int) -> Tuple[int, int]:
    gs = grid_size - 1
    x = int(np.floor(float(xn) * gs + 1e-9))
    y = int(np.floor(float(yn) * gs + 1e-9))
    x = max(0, min(gs, x))
    y = max(0, min(gs, y))
    return x, y

def obs_to_robot_xy(obs: np.ndarray, grid_size: int) -> Tuple[int, int]:
    return norm_xy_to_grid(float(obs[0]), float(obs[1]), grid_size)

def infer_action_mapping_local(seed: int = 0) -> Dict[str, int]:
    e = make_env(RCConfig(seed=int(seed)))
    e.reset(seed=int(seed))

    state0 = {
        "rx": int(e.rx), "ry": int(e.ry),
        "px": int(e.px), "py": int(e.py),
        "dx": int(e.dx), "dy": int(e.dy),
        "cx": int(e.cx), "cy": int(e.cy),
        "battery": int(e.battery),
        "has_package": bool(e.has_package),
    }

    mapping: Dict[str, int] = {}
    for a in range(e.action_space.n):
        e.reset(seed=int(seed))
        force_world_state_on_env(e, state0)
        base = unwrap_env(e)
        obs0 = base._obs() if hasattr(base, "_obs") else None
        if obs0 is None:
            raise RuntimeError("Local env does not expose _obs(); cannot infer action map.")

        x0, y0 = obs_to_robot_xy(np.asarray(obs0, dtype=np.float32), int(e.grid_size))
        obs1, _, _, _, _ = e.step(int(a))
        x1, y1 = obs_to_robot_xy(np.asarray(obs1, dtype=np.float32), int(e.grid_size))

        dx, dy = x1 - x0, y1 - y0
        if (dx, dy) == (0, 1):
            mapping["up"] = a
        elif (dx, dy) == (1, 0):
            mapping["right"] = a
        elif (dx, dy) == (0, -1):
            mapping["down"] = a
        elif (dx, dy) == (-1, 0):
            mapping["left"] = a

    if not all(k in mapping for k in ["up", "right", "down", "left"]):
        raise RuntimeError(f"Could not infer full action map, got: {mapping}")
    return mapping

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

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# =========================
# A5 movement policy (risk-aware charging detour)
# FIX: battery threshold in consistent units
# =========================
def a5_action(
    obs: np.ndarray,
    grid_size: int,
    act_map: Dict[str, int],
    battery_max: int,
    safety_steps: int,
    safety_mult: float,
) -> int:
    has = float(obs[6]) > 0.5
    bat_frac = float(obs[7])  # normalized [0,1]

    rx, ry = obs_to_robot_xy(obs, grid_size)
    px, py = norm_xy_to_grid(float(obs[2]), float(obs[3]), grid_size)
    dx, dy = norm_xy_to_grid(float(obs[4]), float(obs[5]), grid_size)
    cx, cy = norm_xy_to_grid(float(obs[8]), float(obs[9]), grid_size)

    curr = (rx, ry)
    tgt = (dx, dy) if has else (px, py)
    chg = (cx, cy)

    dist_to_tgt = manhattan(curr, tgt)

    # battery need as fraction of battery_max (conservative proxy)
    need_steps = float(dist_to_tgt + int(safety_steps))
    need_frac = (need_steps / max(1.0, float(battery_max))) * float(safety_mult)

    if bat_frac <= need_frac:
        return greedy_action_towards(curr, chg, act_map)
    return greedy_action_towards(curr, tgt, act_map)


# =========================
# Twin builder
# =========================
class EpisodeInitWrapper:
    def __init__(self, env_obj: Any, init_state: Dict[str, Any]):
        self.env = env_obj
        self.init_state = init_state
        self.action_space = env_obj.action_space
        self.observation_space = getattr(env_obj, "observation_space", None)

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        force_world_state_on_env(self.env, self.init_state)
        base = unwrap_env(self.env)
        if hasattr(base, "_obs") and callable(getattr(base, "_obs")):
            obs = base._obs()
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: Any):
        return self.env.step(action)

def make_twin_env(theta: Dict[str, Any], init_state: Dict[str, Any], seed: int) -> Any:
    cfg = RCConfig(
        grid_size=int(theta["grid_size"]),
        battery_max=int(theta["battery_max"]),
        step_cost=float(theta["step_cost"]),
        delivery_reward=float(theta["delivery_reward"]),
        battery_fail_penalty=float(theta["battery_fail_penalty"]),
        use_stay=bool(theta["use_stay"]),
        seed=int(seed),
    )
    base = make_env(cfg)
    epi = EpisodeInitWrapper(base, init_state=init_state)
    twin = SlipActionWrapper(epi, p_slip=float(theta["p_slip"]), seed=int(seed))
    return twin


# =========================
# Replay fixed actions in twin (for calibration)
# =========================
def replay_episode(theta: Dict[str, Any], init_state: Dict[str, Any], seed: int, actions: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    env = make_twin_env(theta, init_state=init_state, seed=int(seed))
    obs, _ = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32)

    obs_list = [obs.copy()]
    r_list: List[float] = []

    for a in actions:
        obs, r, term, trunc, _ = env.step(int(a))
        obs = np.asarray(obs, dtype=np.float32)
        obs_list.append(obs.copy())
        r_list.append(float(r))
        if term or trunc:
            break

    return np.stack(obs_list, axis=0).astype(np.float32), np.asarray(r_list, dtype=float)


# =========================
# Metrics (oracle vs twin) aligned on queried indices
# =========================
def compute_metrics_sparse(
    obs_o: np.ndarray,
    r_o: np.ndarray,
    obs_t: np.ndarray,
    r_t: np.ndarray,
    query_idx_obs: List[int],
) -> Dict[str, float]:
    idx = np.asarray(query_idx_obs, dtype=int)
    idx = idx[(idx >= 0) & (idx < len(obs_t))]

    T = min(len(obs_o), len(idx))
    if T <= 1:
        return {"loss_total": 1e9, "pos_mse": 1e9, "bat_mse": 1e9, "rew_gap": 1e9}

    o = obs_o[:T]
    t = obs_t[idx[:T]]

    pos_mse = float(np.mean((o[:, 0:2] - t[:, 0:2]) ** 2))
    bat_mse = float(np.mean((o[:, 7] - t[:, 7]) ** 2))

    # rewards alignment for queried transitions:
    # idx includes 0 for reset obs; transitions correspond to idx[1:]-1 in r_t
    if T >= 2 and len(r_t) > 0 and len(r_o) >= (T - 1):
        rw_idx = np.clip(idx[1:T] - 1, 0, max(0, len(r_t) - 1))
        rew_gap = float(abs(np.sum(r_o[:(T - 1)]) - np.sum(r_t[rw_idx])))
    else:
        Tr = min(len(r_o), len(r_t))
        rew_gap = float(abs(np.sum(r_o[:Tr]) - np.sum(r_t[:Tr])))

    loss_total = pos_mse + bat_mse + 0.1 * rew_gap
    return {"loss_total": float(loss_total), "pos_mse": pos_mse, "bat_mse": bat_mse, "rew_gap": rew_gap}


def theta_mean_abs(theta_est: Dict[str, Any], ws_true: Dict[str, Any]) -> float:
    keys = ["battery_max", "step_cost", "delivery_reward", "battery_fail_penalty", "p_slip"]
    diffs = []
    for k in keys:
        if k in theta_est and k in ws_true and ws_true[k] is not None:
            diffs.append(abs(float(theta_est[k]) - float(ws_true[k])))
    return float(np.mean(diffs)) if diffs else float("nan")


# =========================
# Random search around current theta
# =========================
def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 400))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.50))
    th["p_slip"] = float(np.clip(float(center["p_slip"]) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.30))
    th["delivery_reward"] = float(center["delivery_reward"])
    th["battery_fail_penalty"] = float(center["battery_fail_penalty"])
    return th


def calibrate_on_episode_actions(
    rng: np.random.Generator,
    theta0: Dict[str, Any],
    init_state: Dict[str, Any],
    seed: int,
    actions: List[int],
    obs_oracle_q: np.ndarray,
    r_oracle_q: np.ndarray,
    query_idx_obs: List[int],
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    obs0, r0 = replay_episode(theta0, init_state=init_state, seed=int(seed), actions=actions)
    m_before = compute_metrics_sparse(obs_oracle_q, r_oracle_q, obs0, r0, query_idx_obs)

    best_theta = dict(theta0)
    best_m = dict(m_before)

    for _ in range(int(n_trials)):
        th = sample_theta(rng, best_theta)
        obs_t, r_t = replay_episode(th, init_state=init_state, seed=int(seed), actions=actions)
        m = compute_metrics_sparse(obs_oracle_q, r_oracle_q, obs_t, r_t, query_idx_obs)
        if m["loss_total"] < best_m["loss_total"]:
            best_theta, best_m = dict(th), dict(m)

    return best_theta, best_m, m_before


# =========================
# Main (SPARSE oracle queries; stable with your current server)
# =========================
def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H_TOTAL = 50_000
    K_TOTAL = int(np.ceil(H_TOTAL / 5))  # keep your original for A5

    EP_MAX_STEPS = 400
    QUERY_EVERY = 25
    CALIB_TRIALS = 300
    SLEEP_S = 0.08

    SAFETY_STEPS = 6
    SAFETY_MULT = 1.4

    SEED0 = 0

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a5_run_{run_id}.jsonl"

    oracle = RoboCourierOracle(ORACLE_URL, timeout_s=30.0, max_retries=6)
    act_map = infer_action_mapping_local(seed=SEED0)
    rng = np.random.default_rng(0)

    theta: Dict[str, Any] = {
        "grid_size": 10,
        "use_stay": False,
        "battery_max": 80,
        "step_cost": 0.10,
        "delivery_reward": 10.0,
        "battery_fail_penalty": 8.0,
        "p_slip": 0.10,
    }

    total_twin_steps = 0
    k_used_total = 0
    ep = 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "start",
            "run_id": run_id,
            "log_mode": "train",
            "baseline": "A5_riskaware_charge_querycalib_scaled",
            "oracle_url": ORACLE_URL,
            "H_TOTAL": int(H_TOTAL),
            "K_TOTAL": int(K_TOTAL),
            "EP_MAX_STEPS": int(EP_MAX_STEPS),
            "QUERY_EVERY": int(QUERY_EVERY),
            "CALIB_TRIALS": int(CALIB_TRIALS),
            "SAFETY_STEPS": int(SAFETY_STEPS),
            "SAFETY_MULT": float(SAFETY_MULT),
            "SLEEP_S": float(SLEEP_S),
            "init_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep

            reset = oracle.reset(seed=int(seed), config_overrides=None)
            sid = reset["session_id"]
            ws0 = reset["world_state"]
            init_state = ws0

            # sync structural params from oracle reset
            theta["grid_size"] = int(ws0.get("grid_size", theta["grid_size"]))
            theta["use_stay"] = bool(ws0.get("use_stay", theta["use_stay"]))
            grid_size = int(theta["grid_size"])

            # episode start log
            f.write(json.dumps({
                "event": "episode_start",
                "run_id": run_id,
                "log_mode": "train",
                "baseline": "A5",
                "ep": int(ep),
                "seed": int(seed),
                "t_global": int(total_twin_steps),
                "k_used_total": int(k_used_total),
                "k_used_ep": 0,
                "theta_est": dict(theta),
                "theta_true_eval_only": {
                    "battery_max": ws0.get("battery_max", None),
                    "step_cost": ws0.get("step_cost", None),
                    "delivery_reward": ws0.get("delivery_reward", None),
                    "battery_fail_penalty": ws0.get("battery_fail_penalty", None),
                    "p_slip": ws0.get("p_slip", None),
                },
            }, ensure_ascii=False) + "\n")

            env = make_twin_env(theta, init_state=init_state, seed=int(seed))
            obs_t, _ = env.reset(seed=int(seed))
            obs_t = np.asarray(obs_t, dtype=np.float32)

            actions: List[int] = []

            # Sparse oracle samples (queried only)
            obs_oracle_q: List[np.ndarray] = [np.asarray(reset["obs"], dtype=np.float32)]
            r_oracle_q: List[float] = []
            query_idx_obs: List[int] = [0]  # align to replay obs indexing

            # per-episode query counter
            k_used_ep = 0

            # running no-GT stats at queried points (oracle obs vs twin obs at same queried time)
            q_n = 0
            pos_se_sum = 0.0
            bat_se_sum = 0.0
            rew_ae_sum = 0.0

            steps_in_ep = 0
            term = False
            trunc = False

            # if we got oracle ws at reset, keep it; later queried steps will update it
            ws_last: Dict[str, Any] = ws0 if isinstance(ws0, dict) else {}

            while (not term) and (not trunc) and (steps_in_ep < EP_MAX_STEPS) and (total_twin_steps < H_TOTAL):
                a = a5_action(
                    obs=obs_t,
                    grid_size=grid_size,
                    act_map=act_map,
                    battery_max=int(theta["battery_max"]),
                    safety_steps=int(SAFETY_STEPS),
                    safety_mult=float(SAFETY_MULT),
                )

                # twin step ALWAYS
                obs_t2, r_twin, term, trunc, _ = env.step(int(a))
                obs_t2 = np.asarray(obs_t2, dtype=np.float32)
                obs_t = obs_t2
                actions.append(int(a))

                # Decide whether to query oracle THIS step
                do_query = (k_used_total < K_TOTAL) and (steps_in_ep % QUERY_EVERY == 0)

                oracle_queried = 0
                query_cost = 0.0
                theta_true_eval_only = None
                acc_eval = None

                # Query oracle only at scheduled points (sparse)
                if do_query:
                    step_o = oracle.step(sid, int(a))
                    oracle_queried = 1
                    query_cost = 1.0
                    k_used_total += 1
                    k_used_ep += 1

                    obs_o = np.asarray(step_o["obs"], dtype=np.float32)
                    r_o = float(step_o.get("reward", 0.0))
                    ws_step = step_o.get("world_state", None)
                    if isinstance(ws_step, dict):
                        ws_last = ws_step

                    obs_oracle_q.append(obs_o)
                    r_oracle_q.append(r_o)
                    query_idx_obs.append(steps_in_ep + 1)

                    # no-GT stats comparing oracle obs to twin obs at same time
                    pos_se = float(np.mean((obs_o[0:2] - obs_t2[0:2]) ** 2))
                    bat_se = float((obs_o[7] - obs_t2[7]) ** 2)
                    rew_ae = float(abs(r_o - float(r_twin)))

                    q_n += 1
                    pos_se_sum += pos_se
                    bat_se_sum += bat_se
                    rew_ae_sum += rew_ae

                    if isinstance(ws_last, dict) and len(ws_last) > 0:
                        th_err = theta_mean_abs(theta, ws_last)
                        if np.isfinite(th_err):
                            acc_eval = {"acc/theta_mean_abs": float(th_err)}
                        theta_true_eval_only = {
                            "battery_max": ws_last.get("battery_max", None),
                            "step_cost": ws_last.get("step_cost", None),
                            "delivery_reward": ws_last.get("delivery_reward", None),
                            "battery_fail_penalty": ws_last.get("battery_fail_penalty", None),
                            "p_slip": ws_last.get("p_slip", None),
                        }

                    if SLEEP_S > 0:
                        time.sleep(float(SLEEP_S))

                acc_no_gt = None
                if q_n > 0:
                    acc_no_gt = {
                        "acc/pos_mse": float(pos_se_sum / q_n),
                        "acc/bat_mse": float(bat_se_sum / q_n),
                        "acc/rew_mae": float(rew_ae_sum / q_n),
                    }

                f.write(json.dumps({
                    "event": "step",
                    "run_id": run_id,
                    "log_mode": "train",
                    "baseline": "A5",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(total_twin_steps),
                    "t_ep": int(steps_in_ep),
                    "action": int(a),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(k_used_ep),
                    "oracle_queried": int(oracle_queried),
                    "query_cost": float(query_cost),
                    "theta_est": dict(theta),
                    "theta_true_eval_only": theta_true_eval_only,
                    "acc_no_gt": acc_no_gt,
                    "acc_eval": acc_eval,
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                total_twin_steps += 1

            # =========================
            # Calibration update (end of episode) on sparse oracle samples
            # =========================
            if len(obs_oracle_q) >= 2 and len(actions) > 0:
                obs_oracle_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                r_oracle_np = np.asarray(r_oracle_q, dtype=float)

                best_theta, m_after, m_before = calibrate_on_episode_actions(
                    rng=rng,
                    theta0=dict(theta),
                    init_state=init_state,
                    seed=int(seed),
                    actions=actions,
                    obs_oracle_q=obs_oracle_np,
                    r_oracle_q=r_oracle_np,
                    query_idx_obs=query_idx_obs,
                    n_trials=int(CALIB_TRIALS),
                )
                theta = dict(best_theta)

                acc_eval_after = None
                if isinstance(ws_last, dict) and len(ws_last) > 0:
                    th_err_after = theta_mean_abs(theta, ws_last)
                    if np.isfinite(th_err_after):
                        acc_eval_after = {"acc/theta_mean_abs": float(th_err_after)}

                f.write(json.dumps({
                    "event": "oracle_update",
                    "run_id": run_id,
                    "log_mode": "train",
                    "baseline": "A5",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(total_twin_steps - 1),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(k_used_ep),
                    "theta_est": dict(theta),
                    "theta_true_eval_only": {
                        "battery_max": ws_last.get("battery_max", None) if isinstance(ws_last, dict) else None,
                        "step_cost": ws_last.get("step_cost", None) if isinstance(ws_last, dict) else None,
                        "delivery_reward": ws_last.get("delivery_reward", None) if isinstance(ws_last, dict) else None,
                        "battery_fail_penalty": ws_last.get("battery_fail_penalty", None) if isinstance(ws_last, dict) else None,
                        "p_slip": ws_last.get("p_slip", None) if isinstance(ws_last, dict) else None,
                    },
                    "acc_eval": acc_eval_after,
                    "loss": {
                        "loss/pos": float(m_after["pos_mse"]),
                        "loss/bat": float(m_after["bat_mse"]),
                        "loss/rew": float(m_after["rew_gap"]),
                        "loss/term": 0.0,
                        "loss/total": float(m_after["loss_total"]),
                    },
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(m_after),
                    "query_idx_obs": list(map(int, query_idx_obs)),
                }, ensure_ascii=False) + "\n")

            ep += 1

        f.write(json.dumps({
            "event": "end",
            "run_id": run_id,
            "log_mode": "train",
            "baseline": "A5",
            "total_twin_steps": int(total_twin_steps),
            "k_used_total": int(k_used_total),
            "final_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

    print("WROTE:", log_path)


if __name__ == "__main__":
    main()

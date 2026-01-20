from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from env import RCConfig, make_env
from rc_calib.wrappers import SlipActionWrapper


# Oracle client
class RoboCourierOracle:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def reset(self, seed: int = 0, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"seed": int(seed), "config_overrides": config_overrides}
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, action: int) -> Dict[str, Any]:
        payload = {"session_id": session_id, "action": int(action)}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


# coords + action mapping

def norm_xy_to_grid(xn: float, yn: float, grid_size: int) -> Tuple[int, int]:
    gs = grid_size - 1
    x = int(round(float(xn) * gs))
    y = int(round(float(yn) * gs))
    x = max(0, min(gs, x))
    y = max(0, min(gs, y))
    return x, y

def obs_to_robot_xy(obs: np.ndarray, grid_size: int) -> Tuple[int, int]:
    return norm_xy_to_grid(float(obs[0]), float(obs[1]), grid_size)

def infer_action_mapping_local(seed: int = 0) -> Dict[str, int]:
    e = make_env(RCConfig(seed=int(seed)))
    obs0, _ = e.reset(seed=int(seed))
    x0, y0 = obs_to_robot_xy(np.asarray(obs0, dtype=np.float32), int(e.grid_size))

    mapping: Dict[str, int] = {}
    for a in range(e.action_space.n):
        e.reset(seed=int(seed))
        obs1, _, _, _, _ = e.step(int(a))
        x1, y1 = obs_to_robot_xy(np.asarray(obs1, dtype=np.float32), int(e.grid_size))
        dx, dy = x1 - x0, y1 - y0
        if (dx, dy) == (0, 1): mapping["up"] = a
        elif (dx, dy) == (1, 0): mapping["right"] = a
        elif (dx, dy) == (0, -1): mapping["down"] = a
        elif (dx, dy) == (-1, 0): mapping["left"] = a

    if len(mapping) < 4:
        raise RuntimeError(f"Could not infer full action map, got: {mapping}")
    return mapping

def greedy_action_towards(curr: Tuple[int, int], target: Tuple[int, int], act: Dict[str, int]) -> int:
    cx, cy = curr
    tx, ty = target
    if cx < tx: return act["right"]
    if cx > tx: return act["left"]
    if cy < ty: return act["up"]
    if cy > ty: return act["down"]
    return act["up"]


# battery-aware greedy (A3 core)

def a3_policy_action_grid(obs: np.ndarray, grid_size: int, act_map: Dict[str, int], charge_thresh: float) -> int:
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


# Twin builder
def make_twin_env(theta: Dict[str, Any], seed: int) -> Any:
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
    return SlipActionWrapper(base, p_slip=float(theta.get("p_slip", 0.0)), seed=int(seed))


# Theta accuracy vs oracle world_state 
def theta_mean_abs(theta: Dict[str, Any], ws: Dict[str, Any]) -> float:
    keys = ["battery_max", "step_cost", "delivery_reward", "battery_fail_penalty"]
    diffs = []
    for k in keys:
        if k in theta and k in ws:
            diffs.append(abs(float(theta[k]) - float(ws[k])))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


# Chunk calibration replay actions and optimize theta
def replay_local(env_obj: Any, actions: List[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    obs, _ = env_obj.reset(seed=int(seed))
    obs_list = [np.asarray(obs, dtype=np.float32)]
    r_list: List[float] = []
    for a in actions:
        obs, r, term, trunc, _ = env_obj.step(int(a))
        obs_list.append(np.asarray(obs, dtype=np.float32))
        r_list.append(float(r))
        if term or trunc:
            break
    return np.stack(obs_list, axis=0), np.asarray(r_list, dtype=float)

def compute_metrics(obs_p: np.ndarray, r_p: np.ndarray, obs_t: np.ndarray, r_t: np.ndarray) -> Dict[str, float]:
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))
    if T <= 1:
        return {"loss": 1e9, "pos_mse": 1e9, "bat_mse": 1e9, "rew_gap": 1e9}

    pos_mse = float(np.mean((obs_p[:T, 0:2] - obs_t[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_p[:Tr]) - np.sum(r_t[:Tr])))

    loss = pos_mse + bat_mse + 0.1 * rew_gap
    return {"loss": float(loss), "pos_mse": pos_mse, "bat_mse": bat_mse, "rew_gap": rew_gap}

def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 400))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.50))
    th["delivery_reward"] = float(np.clip(float(center["delivery_reward"]) + float(rng.uniform(-6.0, 6.0)), 1.0, 20.0))
    th["battery_fail_penalty"] = float(np.clip(float(center["battery_fail_penalty"]) + float(rng.uniform(-8.0, 8.0)), 0.0, 25.0))
    th["p_slip"] = float(np.clip(float(center.get("p_slip", 0.0)) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.30))
    return th

def calibrate_on_chunk(
    rng: np.random.Generator,
    base_theta: Dict[str, Any],
    seed: int,
    actions: List[int],
    obs_oracle: np.ndarray,
    r_oracle: np.ndarray,
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    # BEFORE
    tw0 = make_twin_env(base_theta, seed=int(seed))
    obs0, r0 = replay_local(tw0, actions, seed=int(seed))
    m_before = compute_metrics(obs_oracle, r_oracle, obs0, r0)

    best_theta = dict(base_theta)
    best_m = dict(m_before)

    for _ in range(int(n_trials)):
        th = sample_theta(rng, best_theta)
        tw = make_twin_env(th, seed=int(seed))
        obs_t, r_t = replay_local(tw, actions, seed=int(seed))
        m = compute_metrics(obs_oracle, r_oracle, obs_t, r_t)
        if m["loss"] < best_m["loss"]:
            best_theta, best_m = dict(th), dict(m)

    return best_theta, best_m, m_before


def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H_TOTAL = 50_000
    K_TOTAL = int(np.ceil(H_TOTAL / 5))

    # A4-style chunks
    K_EP = 120            
    EP_MAX_STEPS = 400    
    CALIB_TRIALS = 300    
    SLEEP_S = 0.08

    CHARGE_THRESH = 0.30
    SEED0 = 0

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a3c_run_{run_id}.jsonl" 

    oracle = RoboCourierOracle(ORACLE_URL)
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
            "baseline": "A3_calib_chunks(A4_style)",
            "oracle_url": ORACLE_URL,
            "H_TOTAL": H_TOTAL,
            "K_TOTAL": K_TOTAL,
            "K_EP": K_EP,
            "EP_MAX_STEPS": EP_MAX_STEPS,
            "CALIB_TRIALS": CALIB_TRIALS,
            "SLEEP_S": SLEEP_S,
            "CHARGE_THRESH": CHARGE_THRESH,
            "init_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep
            reset = oracle.reset(seed=int(seed))
            sid = reset["session_id"]
            ws0 = reset["world_state"]

            # sync static properties from oracle 
            theta["grid_size"] = int(ws0.get("grid_size", theta["grid_size"]))
            theta["use_stay"] = bool(ws0.get("use_stay", theta["use_stay"]))
            grid_size = int(theta["grid_size"])

            env = make_twin_env(theta, seed=int(seed))
            obs_t, _ = env.reset(seed=int(seed))
            obs_t = np.asarray(obs_t, dtype=np.float32)

            # chunk buffers 
            actions_q: List[int] = []
            obs_oracle_q: List[np.ndarray] = [np.asarray(reset["obs"], dtype=np.float32)]
            r_oracle_q: List[float] = []

            steps_in_ep = 0
            term = False
            trunc = False

            # episode oracle budget
            k_ep = min(int(K_EP), max(0, int(K_TOTAL - k_used_total)))
            oracle_active = (k_ep > 0)

            while (not term) and (not trunc) and steps_in_ep < EP_MAX_STEPS and total_twin_steps < H_TOTAL:
                a = a3_policy_action_grid(obs_t, grid_size, act_map, CHARGE_THRESH)

                obs_t, r_t, term, trunc, _ = env.step(int(a))
                obs_t = np.asarray(obs_t, dtype=np.float32)

                oracle_queried = 0
                ws_step = ws0

                if oracle_active and (len(actions_q) < k_ep) and (k_used_total < K_TOTAL):
                    step_o = oracle.step(sid, int(a))
                    k_used_total += 1
                    oracle_queried = 1

                    ws_step = step_o.get("world_state", ws0)
                    obs_oracle_q.append(np.asarray(step_o["obs"], dtype=np.float32))
                    r_oracle_q.append(float(step_o["reward"]))
                    actions_q.append(int(a))

                    if step_o.get("terminated") or step_o.get("truncated"):
                        oracle_active = False

                    if SLEEP_S > 0:
                        time.sleep(float(SLEEP_S))

                # theta error для overlay
                th_err = theta_mean_abs(theta, ws_step)

                f.write(json.dumps({
                    "event": "step",
                    "run_id": run_id,
                    "baseline": "A3C",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(total_twin_steps),
                    "t_ep": int(steps_in_ep),
                    "k_used_total": int(k_used_total),
                    "oracle_queried": int(oracle_queried),
                    "acc": {"acc/theta_mean_abs": float(th_err) if np.isfinite(th_err) else None},
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                total_twin_steps += 1

                if oracle_active and len(actions_q) >= k_ep:
                    oracle_active = False

            #calib update at episode end 
            if len(actions_q) > 0 and len(obs_oracle_q) >= 2:
                obs_oracle_q_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                r_oracle_q_np = np.asarray(r_oracle_q, dtype=float)

                best_theta, best_m, m_before = calibrate_on_chunk(
                    rng=rng,
                    base_theta=dict(theta),
                    seed=int(seed),
                    actions=actions_q,
                    obs_oracle=obs_oracle_q_np,
                    r_oracle=r_oracle_q_np,
                    n_trials=int(CALIB_TRIALS),
                )

                theta = dict(best_theta)

                # accuracy after update 
                th_err_after = theta_mean_abs(theta, ws0)

                f.write(json.dumps({
                    "event": "calib_update",
                    "run_id": run_id,
                    "baseline": "A3C",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(total_twin_steps),
                    "k_used_total": int(k_used_total),
                    "k_ep": int(k_ep),
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(best_m),
                    "new_theta": dict(best_theta),
                    "acc": {"acc/theta_mean_abs": float(th_err_after) if np.isfinite(th_err_after) else None},
                }, ensure_ascii=False) + "\n")

            ep += 1

        f.write(json.dumps({
            "event": "end",
            "run_id": run_id,
            "total_twin_steps": int(total_twin_steps),
            "k_used_total": int(k_used_total),
            "final_theta": dict(theta),
            "calib_updates": int(ep),  
        }, ensure_ascii=False) + "\n")

    print(f"WROTE: {log_path}")


if __name__ == "__main__":
    main()

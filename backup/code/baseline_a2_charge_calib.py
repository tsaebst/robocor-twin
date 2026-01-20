# ============================================
# baseline_a1_scaled.py  (A1 @ A5/PPO scale)
# - Runs for H_TOTAL twin steps (e.g., 50k)
# - Uses oracle budget K_TOTAL and query schedule QUERY_EVERY
# - Logs step-by-step: t_global, k_used_total, queried_oracle
# - Performs per-episode calibration update using ONLY queried points
# - Logs calib_update with acc/theta_mean_abs for overlay plots
# ============================================

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


# --------------------------
# Oracle client
# --------------------------
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


# --------------------------
# Helpers: unwrap + state set
# --------------------------
def unwrap_env(env_obj: Any, max_depth: int = 20) -> Any:
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


class EpisodeInitWrapper:
    """Force per-episode init_state after reset()."""
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


# --------------------------
# Action mapping + coords
# --------------------------
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


def a1_action(obs_twin: np.ndarray, grid_size: int, act_map: Dict[str, int]) -> int:
    has = float(obs_twin[6]) > 0.5
    rx, ry = obs_to_robot_xy(obs_twin, grid_size)
    if not has:
        px, py = norm_xy_to_grid(float(obs_twin[2]), float(obs_twin[3]), grid_size)
        tgt = (px, py)
    else:
        dx, dy = norm_xy_to_grid(float(obs_twin[4]), float(obs_twin[5]), grid_size)
        tgt = (dx, dy)
    return greedy_action_towards((rx, ry), tgt, act_map)


# --------------------------
# Metrics on queried points
# --------------------------
def _take_at_indices(arr: np.ndarray, idx: List[int]) -> np.ndarray:
    if arr.size == 0 or not idx:
        return np.zeros((0, arr.shape[1]), dtype=np.float32) if arr.ndim == 2 else np.zeros((0,), dtype=float)
    ii = np.asarray(idx, dtype=int)
    ii = ii[(ii >= 0) & (ii < len(arr))]
    return arr[ii]


def compute_loss_on_queries(obs_oracle_q: np.ndarray, obs_twin: np.ndarray, query_steps: List[int]) -> Dict[str, float]:
    # Compare only at query steps (positions + battery as main signals)
    tq = _take_at_indices(obs_twin, query_steps)
    pq = obs_oracle_q  # already aligned as "one obs per query", same order as query_steps

    T = min(len(pq), len(tq))
    if T <= 1:
        return {"loss": 1e9, "pos_mse": 1e9, "bat_mse": 1e9}

    pos_mse = float(np.mean((pq[:T, 0:2] - tq[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((pq[:T, 7] - tq[:T, 7]) ** 2))
    loss = pos_mse + bat_mse
    return {"loss": float(loss), "pos_mse": pos_mse, "bat_mse": bat_mse}


def theta_mean_abs(theta: Dict[str, Any], ws_true: Dict[str, Any]) -> float:
    # robust to missing keys; skip those absent
    keys = ["battery_max", "step_cost", "delivery_reward", "battery_fail_penalty", "p_slip"]
    diffs = []
    for k in keys:
        if k in ws_true and k in theta:
            diffs.append(abs(float(theta[k]) - float(ws_true[k])))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


# --------------------------
# Calibration: random search around current theta
# --------------------------
def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 600))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.80))
    th["delivery_reward"] = float(np.clip(float(center["delivery_reward"]) + float(rng.uniform(-6.0, 6.0)), 1.0, 30.0))
    th["battery_fail_penalty"] = float(np.clip(float(center["battery_fail_penalty"]) + float(rng.uniform(-8.0, 8.0)), 0.0, 40.0))
    th["p_slip"] = float(np.clip(float(center["p_slip"]) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.35))
    return th


def replay_episode(theta: Dict[str, Any], init_state: Dict[str, Any], seed: int, actions: List[int]) -> np.ndarray:
    env = make_twin_env(theta, init_state=init_state, seed=seed)
    obs, _ = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32)
    obs_list = [obs.copy()]
    for a in actions:
        obs, _, term, trunc, _ = env.step(int(a))
        obs = np.asarray(obs, dtype=np.float32)
        obs_list.append(obs.copy())
        if term or trunc:
            break
    return np.stack(obs_list, axis=0).astype(np.float32)


def calibrate_episode(
    rng: np.random.Generator,
    theta0: Dict[str, Any],
    init_state: Dict[str, Any],
    seed: int,
    actions: List[int],
    obs_oracle_q: np.ndarray,
    query_steps: List[int],
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    obs_t0 = replay_episode(theta0, init_state, seed, actions)
    best_m = compute_loss_on_queries(obs_oracle_q, obs_t0, query_steps)
    best_theta = dict(theta0)

    for _ in range(int(n_trials)):
        th = sample_theta(rng, best_theta)
        obs_t = replay_episode(th, init_state, seed, actions)
        m = compute_loss_on_queries(obs_oracle_q, obs_t, query_steps)
        if m["loss"] < best_m["loss"]:
            best_m = dict(m)
            best_theta = dict(th)

    return best_theta, best_m


# --------------------------
# MAIN (scaled run)
# --------------------------
def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    # Match PPO scale here:
    H_TOTAL = 50_000            # total twin steps (set to 2000 for quick debug)
    K_TOTAL = 2_000             # oracle budget (set to match PPO run or A5 config)
    EP_MAX_STEPS = 400
    QUERY_EVERY = 25
    CALIB_TRIALS = 300
    SLEEP_S = 0.08

    SEED0 = 0

    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a1_run_{run_id}.jsonl"

    oracle = RoboCourierOracle(ORACLE_URL)
    act_map = infer_action_mapping_local(seed=SEED0)
    rng = np.random.default_rng(0)

    # initial theta (will be overwritten by oracle WS per-episode for grid_size/use_stay if needed)
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
    total_oracle_steps = 0
    ep = 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "start",
            "run_id": run_id,
            "baseline": "A1_greedy_PD_no_charge_scaled",
            "oracle_url": ORACLE_URL,
            "H_TOTAL": H_TOTAL,
            "K_TOTAL": K_TOTAL,
            "EP_MAX_STEPS": EP_MAX_STEPS,
            "QUERY_EVERY": QUERY_EVERY,
            "CALIB_TRIALS": CALIB_TRIALS,
            "SLEEP_S": SLEEP_S,
            "init_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep

            reset = oracle.reset(seed=seed, config_overrides=None)
            sid = reset["session_id"]
            ws0 = reset["world_state"]

            # sync structural params
            theta["grid_size"] = int(ws0.get("grid_size", theta["grid_size"]))
            theta["use_stay"] = bool(ws0.get("use_stay", theta["use_stay"]))

            init_state = ws0
            env = make_twin_env(theta, init_state=init_state, seed=seed)
            obs_t, _ = env.reset(seed=seed)
            obs_t = np.asarray(obs_t, dtype=np.float32)

            f.write(json.dumps({
                "event": "episode_start",
                "run_id": run_id,
                "ep": ep,
                "seed": seed,
                "t_global": total_twin_steps,
                "k_used_total": total_oracle_steps,
                "theta": dict(theta),
            }, ensure_ascii=False) + "\n")

            actions: List[int] = []
            query_steps: List[int] = []
            obs_oracle_q: List[np.ndarray] = []

            # If you want to include reset obs as a "query at t=0"
            # (helps stabilize metrics), keep this:
            obs_oracle_q.append(np.asarray(reset["obs"], dtype=np.float32))
            query_steps.append(0)

            steps_in_ep = 0
            term = False
            trunc = False

            while (not term) and (not trunc) and (steps_in_ep < EP_MAX_STEPS) and (total_twin_steps < H_TOTAL):
                a = a1_action(obs_t, grid_size=int(theta["grid_size"]), act_map=act_map)

                # step twin
                obs_t, r_t, term, trunc, _ = env.step(int(a))
                obs_t = np.asarray(obs_t, dtype=np.float32)
                actions.append(int(a))

                queried = False
                if (total_oracle_steps < K_TOTAL) and (steps_in_ep % QUERY_EVERY == 0):
                    step = oracle.step(sid, int(a))
                    total_oracle_steps += 1
                    queried = True
                    obs_oracle_q.append(np.asarray(step["obs"], dtype=np.float32))
                    query_steps.append(steps_in_ep + 1)  # +1 because obs_t already advanced

                    if SLEEP_S > 0:
                        time.sleep(float(SLEEP_S))

                f.write(json.dumps({
                    "event": "step",
                    "run_id": run_id,
                    "ep": ep,
                    "seed": seed,
                    "t_global": total_twin_steps,
                    "t_ep": steps_in_ep,
                    "k_used_total": total_oracle_steps,
                    "queried_oracle": int(queried),
                    "action": int(a),
                    "theta": dict(theta),
                    "obs_twin": obs_t.tolist(),
                    "reward_twin": float(r_t),
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                total_twin_steps += 1

            # calibration update (if we have at least 2 query points)
            if len(obs_oracle_q) >= 2:
                obs_oracle_q_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                theta_before = dict(theta)

                best_theta, best_m = calibrate_episode(
                    rng=rng,
                    theta0=theta_before,
                    init_state=init_state,
                    seed=seed,
                    actions=actions,
                    obs_oracle_q=obs_oracle_q_np,
                    query_steps=query_steps,
                    n_trials=CALIB_TRIALS,
                )
                theta = dict(best_theta)

                # theta accuracy vs prototype (from ws0)
                acc_theta = theta_mean_abs(theta, ws0)

                f.write(json.dumps({
                    "event": "calib_update",
                    "run_id": run_id,
                    "ep": ep,
                    "seed": seed,
                    "t_global": total_twin_steps - 1,
                    "k_used_total": total_oracle_steps,
                    "metrics_after": dict(best_m),
                    "theta_after": dict(theta),
                    "acc": {"acc/theta_mean_abs": float(acc_theta)},
                }, ensure_ascii=False) + "\n")

            ep += 1

        f.write(json.dumps({
            "event": "end",
            "run_id": run_id,
            "total_twin_steps": total_twin_steps,
            "total_oracle_steps": total_oracle_steps,
            "final_theta": theta,
        }, ensure_ascii=False) + "\n")

    print("WROTE:", log_path)


if __name__ == "__main__":
    main()

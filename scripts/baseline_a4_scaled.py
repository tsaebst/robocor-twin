from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from env import RCConfig, make_env
from wrappers import SlipActionWrapper


# =========================
# Oracle client
# =========================
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


# =========================
# Env unwrap + force init world_state (A1/A2-compatible)
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
    # coordinates
    base.rx, base.ry = int(state["rx"]), int(state["ry"])
    base.px, base.py = int(state["px"]), int(state["py"])
    base.dx, base.dy = int(state["dx"]), int(state["dy"])
    base.cx, base.cy = int(state["cx"]), int(state["cy"])
    # dynamics bits
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
    twin = SlipActionWrapper(epi, p_slip=float(theta.get("p_slip", 0.0)), seed=int(seed))
    return twin


# =========================
# Coords + action mapping (robust, A1/A2-style)
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


# =========================
# A4 movement policy (same as A3 “battery-aware greedy”)
# NOTE: calibration policy = chunks, not this.
# =========================
def a4_action(obs: np.ndarray, grid_size: int, act_map: Dict[str, int], charge_thresh: float) -> int:
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


# =========================
# Metrics + calibration (chunk replay)
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

def compute_metrics(obs_oracle: np.ndarray, r_oracle: np.ndarray,
                    obs_twin: np.ndarray, r_twin: np.ndarray) -> Dict[str, float]:
    T = min(len(obs_oracle), len(obs_twin))
    Tr = min(len(r_oracle), len(r_twin))
    if T <= 1:
        return {"loss_total": 1e9, "pos_mse": 1e9, "bat_mse": 1e9, "rew_gap": 1e9}

    pos_mse = float(np.mean((obs_oracle[:T, 0:2] - obs_twin[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((obs_oracle[:T, 7] - obs_twin[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_oracle[:Tr]) - np.sum(r_twin[:Tr])))

    loss_total = pos_mse + bat_mse + 0.1 * rew_gap
    return {"loss_total": float(loss_total), "pos_mse": pos_mse, "bat_mse": bat_mse, "rew_gap": rew_gap}

def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 600))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.80))
    th["delivery_reward"] = float(np.clip(float(center["delivery_reward"]) + float(rng.uniform(-6.0, 6.0)), 1.0, 30.0))
    th["battery_fail_penalty"] = float(np.clip(float(center["battery_fail_penalty"]) + float(rng.uniform(-8.0, 8.0)), 0.0, 40.0))
    th["p_slip"] = float(np.clip(float(center.get("p_slip", 0.10)) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.35))
    return th

def calibrate_on_chunk(
    rng: np.random.Generator,
    theta0: Dict[str, Any],
    init_state: Dict[str, Any],
    seed: int,
    actions: List[int],
    obs_oracle: np.ndarray,
    r_oracle: np.ndarray,
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    obs0, r0 = replay_episode(theta0, init_state=init_state, seed=int(seed), actions=actions)
    m_before = compute_metrics(obs_oracle, r_oracle, obs0, r0)

    best_theta = dict(theta0)
    best_m = dict(m_before)

    for _ in range(int(n_trials)):
        th = sample_theta(rng, best_theta)
        obs_t, r_t = replay_episode(th, init_state=init_state, seed=int(seed), actions=actions)
        m = compute_metrics(obs_oracle, r_oracle, obs_t, r_t)
        if m["loss_total"] < best_m["loss_total"]:
            best_theta, best_m = dict(th), dict(m)

    return best_theta, best_m, m_before


# =========================
# Theta error vs world_state (for eval-only overlay)
# =========================
def theta_mean_abs(theta: Dict[str, Any], ws: Dict[str, Any]) -> float:
    keys = ["battery_max", "step_cost", "delivery_reward", "battery_fail_penalty", "p_slip"]
    diffs = []
    for k in keys:
        if k in theta and k in ws:
            diffs.append(abs(float(theta[k]) - float(ws[k])))
    return float(np.mean(diffs)) if diffs else float("nan")


# =========================
# Main (A4)
# =========================
def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    # Match PPO scale:
    H_TOTAL = 50_000
    K_TOTAL = 2_000

    # A4 chunk policy:
    K_EP = 120                 # number of oracle queries per episode (front-loaded)
    EP_MAX_STEPS = 400
    CALIB_TRIALS = 300
    CHARGE_THRESH = 0.30
    SLEEP_S = 0.08

    SEED0 = 0

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a4_run_{run_id}.jsonl"

    oracle = RoboCourierOracle(ORACLE_URL)
    act_map = infer_action_mapping_local(seed=SEED0)
    rng = np.random.default_rng(0)

    # initial theta
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
        # START
        f.write(json.dumps({
            "event": "start",
            "run_id": run_id,
            "log_mode": "train",
            "baseline": "A4_chunk_oracle_then_calib_scaled",
            "oracle_url": ORACLE_URL,
            "H_TOTAL": int(H_TOTAL),
            "K_TOTAL": int(K_TOTAL),
            "K_EP": int(K_EP),
            "EP_MAX_STEPS": int(EP_MAX_STEPS),
            "CALIB_TRIALS": int(CALIB_TRIALS),
            "CHARGE_THRESH": float(CHARGE_THRESH),
            "SLEEP_S": float(SLEEP_S),
            "init_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep

            reset = oracle.reset(seed=int(seed), config_overrides=None)
            sid = reset["session_id"]
            ws0 = reset["world_state"]

            # sync structural params
            theta["grid_size"] = int(ws0.get("grid_size", theta["grid_size"]))
            theta["use_stay"] = bool(ws0.get("use_stay", theta["use_stay"]))
            grid_size = int(theta["grid_size"])

            init_state = ws0

            env = make_twin_env(theta, init_state=init_state, seed=int(seed))
            obs_t, _ = env.reset(seed=int(seed))
            obs_t = np.asarray(obs_t, dtype=np.float32)

            # per-episode oracle budget
            k_ep_budget = min(int(K_EP), max(0, int(K_TOTAL - k_used_total)))
            oracle_active = (k_ep_budget > 0)

            # buffers for chunk calibration (only queried segment)
            actions_q: List[int] = []
            obs_oracle_q: List[np.ndarray] = [np.asarray(reset["obs"], dtype=np.float32)]
            r_oracle_q: List[float] = []  # rewards returned by oracle.step

            # running no-GT stats on query points (for plot 4)
            q_n = 0
            pos_se_sum = 0.0
            bat_se_sum = 0.0
            rew_ae_sum = 0.0

            # EP START
            f.write(json.dumps({
                "event": "episode_start",
                "run_id": run_id,
                "log_mode": "train",
                "baseline": "A4",
                "ep": int(ep),
                "seed": int(seed),
                "t_global": int(total_twin_steps),
                "k_used_total": int(k_used_total),
                "k_used_ep": 0,
                "theta_est": dict(theta),
            }, ensure_ascii=False) + "\n")

            steps_in_ep = 0
            term = False
            trunc = False

            while (not term) and (not trunc) and (steps_in_ep < EP_MAX_STEPS) and (total_twin_steps < H_TOTAL):
                a = a4_action(obs_t, grid_size=grid_size, act_map=act_map, charge_thresh=CHARGE_THRESH)

                # step twin
                obs_t2, r_twin, term, trunc, _ = env.step(int(a))
                obs_t2 = np.asarray(obs_t2, dtype=np.float32)
                obs_t = obs_t2

                oracle_queried = 0
                ws_step = None
                obs_o = None
                r_o = None

                # query oracle only while in chunk budget
                if oracle_active and (len(actions_q) < k_ep_budget) and (k_used_total < K_TOTAL):
                    step_o = oracle.step(sid, int(a))
                    oracle_queried = 1
                    k_used_total += 1

                    ws_step = step_o.get("world_state", None)
                    obs_o = np.asarray(step_o["obs"], dtype=np.float32)
                    r_o = float(step_o.get("reward", 0.0))

                    obs_oracle_q.append(obs_o)
                    r_oracle_q.append(r_o)
                    actions_q.append(int(a))

                    # update running query-point stats
                    # compare oracle obs to twin obs at the same post-step time
                    pos_se = float(np.mean((obs_o[0:2] - obs_t2[0:2]) ** 2))
                    bat_se = float((obs_o[7] - obs_t2[7]) ** 2)
                    rew_ae = float(abs(r_o - float(r_twin)))

                    q_n += 1
                    pos_se_sum += pos_se
                    bat_se_sum += bat_se
                    rew_ae_sum += rew_ae

                    # stop oracle if terminated early
                    if step_o.get("terminated") or step_o.get("truncated"):
                        oracle_active = False

                    if SLEEP_S > 0:
                        time.sleep(float(SLEEP_S))

                if oracle_active and len(actions_q) >= k_ep_budget:
                    oracle_active = False

                k_used_ep = len(actions_q)

                # no-GT operational metrics (available even without ws_true)
                acc_no_gt = None
                if q_n > 0:
                    acc_no_gt = {
                        "acc/pos_mse": float(pos_se_sum / q_n),
                        "acc/bat_mse": float(bat_se_sum / q_n),
                        "acc/rew_mae": float(rew_ae_sum / q_n),
                    }

                # eval-only theta error (only if oracle returned world_state)
                acc_eval = None
                theta_true_eval_only = None
                if ws_step is not None:
                    th_err = theta_mean_abs(theta, ws_step)
                    if np.isfinite(th_err):
                        acc_eval = {"acc/theta_mean_abs": float(th_err)}
                    theta_true_eval_only = {
                        "battery_max": ws_step.get("battery_max", None),
                        "step_cost": ws_step.get("step_cost", None),
                        "delivery_reward": ws_step.get("delivery_reward", None),
                        "battery_fail_penalty": ws_step.get("battery_fail_penalty", None),
                        "p_slip": ws_step.get("p_slip", None),
                    }

                # STEP LOG
                f.write(json.dumps({
                    "event": "step",
                    "run_id": run_id,
                    "log_mode": "train",
                    "baseline": "A4",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(total_twin_steps),
                    "t_ep": int(steps_in_ep),
                    "action": int(a),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(k_used_ep),
                    "oracle_queried": int(oracle_queried),
                    "query_cost": 1.0,  # for plot 9 (simple constant cost)
                    "theta_est": dict(theta),
                    "theta_true_eval_only": theta_true_eval_only,
                    "acc_no_gt": acc_no_gt,
                    "acc_eval": acc_eval,
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                total_twin_steps += 1

            # ================
            # ORACLE UPDATE (calibration) at episode end
            # ================
            if (len(actions_q) > 0) and (len(obs_oracle_q) >= 2):
                obs_oracle_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                r_oracle_np = np.asarray(r_oracle_q, dtype=float)

                best_theta, m_after, m_before = calibrate_on_chunk(
                    rng=rng,
                    theta0=dict(theta),
                    init_state=init_state,
                    seed=int(seed),
                    actions=actions_q,
                    obs_oracle=obs_oracle_np,
                    r_oracle=r_oracle_np,
                    n_trials=int(CALIB_TRIALS),
                )

                theta = dict(best_theta)

                # one more eval theta error vs initial ws0 (available at reset)
                th_err_after = theta_mean_abs(theta, ws0)
                acc_eval_after = {"acc/theta_mean_abs": float(th_err_after)} if np.isfinite(th_err_after) else None

                # UPDATE LOG (name as oracle_update for plotting compatibility)
                f.write(json.dumps({
                    "event": "oracle_update",
                    "run_id": run_id,
                    "log_mode": "train",
                    "baseline": "A4",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(total_twin_steps - 1),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(len(actions_q)),
                    "theta_est": dict(theta),
                    "theta_true_eval_only": {
                        "battery_max": ws0.get("battery_max", None),
                        "step_cost": ws0.get("step_cost", None),
                        "delivery_reward": ws0.get("delivery_reward", None),
                        "battery_fail_penalty": ws0.get("battery_fail_penalty", None),
                        "p_slip": ws0.get("p_slip", None),
                    },
                    "acc_eval": acc_eval_after,

                    # loss decomposition for plot 10 (keys match your loss plots)
                    "loss": {
                        "loss/pos": float(m_after["pos_mse"]),
                        "loss/bat": float(m_after["bat_mse"]),
                        "loss/rew": float(m_after["rew_gap"]),
                        "loss/term": 0.0,
                        "loss/total": float(m_after["loss_total"]),
                    },

                    # detailed before/after (useful for debugging)
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(m_after),
                }, ensure_ascii=False) + "\n")

            ep += 1

        # END
        f.write(json.dumps({
            "event": "end",
            "run_id": run_id,
            "log_mode": "train",
            "baseline": "A4",
            "total_twin_steps": int(total_twin_steps),
            "k_used_total": int(k_used_total),
            "final_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

    print(f"WROTE: {log_path}")


if __name__ == "__main__":
    main()

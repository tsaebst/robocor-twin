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
# unwrap + force world state (as in A1/A2)
# =========================
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


# =========================
# coords + action mapping
# =========================
def norm_xy_to_grid(xn: float, yn: float, grid_size: int) -> Tuple[int, int]:
    # keep consistent with A1/A2 (floor)
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
        if (dx, dy) == (0, 1): mapping["up"] = a
        elif (dx, dy) == (1, 0): mapping["right"] = a
        elif (dx, dy) == (0, -1): mapping["down"] = a
        elif (dx, dy) == (-1, 0): mapping["left"] = a

    if not all(k in mapping for k in ["up", "right", "down", "left"]):
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


# =========================
# A3 movement policy (battery-aware greedy)
# =========================
def a3_action(obs: np.ndarray, grid_size: int, act_map: Dict[str, int], charge_thresh: float) -> int:
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
# Twin builder (with forced init state)
# =========================
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
# Metrics (PPO-style naming)
# =========================
def _take_at_indices(arr: np.ndarray, idx: List[int]) -> np.ndarray:
    if arr.size == 0 or not idx:
        return np.zeros((0, arr.shape[1]), dtype=np.float32) if arr.ndim == 2 else np.zeros((0,), dtype=float)
    ii = np.asarray(idx, dtype=int)
    ii = ii[(ii >= 0) & (ii < len(arr))]
    return arr[ii]


def compute_no_gt_metrics_on_queries(
    obs_oracle_q: np.ndarray,
    rew_oracle_q: np.ndarray,
    obs_twin: np.ndarray,
    rew_twin: np.ndarray,
    query_steps: List[int],
) -> Dict[str, float]:
    # align twin samples at query steps to oracle query observations
    tq = _take_at_indices(obs_twin, query_steps)
    pq = obs_oracle_q
    T = min(len(pq), len(tq))
    if T <= 1:
        return {"pos_mse": 1e9, "bat_mse": 1e9, "rew_mae": 1e9}

    pos_mse = float(np.mean((pq[:T, 0:2] - tq[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((pq[:T, 7] - tq[:T, 7]) ** 2))

    # reward MAE on the same query-aligned steps:
    # oracle rewards correspond to steps that were queried; twin rewards at those steps are rew_twin[step-1]
    # because obs_twin includes t=0 as first obs.
    # query_steps contains indices in obs space; step reward index = query_step-1 (skip t=0 query)
    rr = []
    for i, qs in enumerate(query_steps[:T]):
        if qs <= 0:
            continue
        if (qs - 1) < len(rew_twin) and i < len(rew_oracle_q):
            rr.append(abs(float(rew_oracle_q[i]) - float(rew_twin[qs - 1])))
    rew_mae = float(np.mean(rr)) if len(rr) > 0 else 0.0

    return {"pos_mse": pos_mse, "bat_mse": bat_mse, "rew_mae": rew_mae}


def theta_mean_abs(theta: Dict[str, Any], ws_true: Dict[str, Any]) -> float:
    keys = ["battery_max", "step_cost", "delivery_reward", "battery_fail_penalty", "p_slip"]
    diffs = []
    for k in keys:
        if k in theta and k in ws_true:
            diffs.append(abs(float(theta[k]) - float(ws_true[k])))
    return float(np.mean(diffs)) if diffs else float("nan")


# =========================
# Calibration (random search) on collected chunk
# =========================
def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 400))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.50))
    th["delivery_reward"] = float(np.clip(float(center["delivery_reward"]) + float(rng.uniform(-6.0, 6.0)), 1.0, 20.0))
    th["battery_fail_penalty"] = float(np.clip(float(center["battery_fail_penalty"]) + float(rng.uniform(-8.0, 8.0)), 0.0, 25.0))
    th["p_slip"] = float(np.clip(float(center.get("p_slip", 0.0)) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.30))
    return th


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


def calibrate_on_chunk(
    rng: np.random.Generator,
    theta0: Dict[str, Any],
    init_state: Dict[str, Any],
    seed: int,
    actions: List[int],
    obs_oracle_q: np.ndarray,
    rew_oracle_q: np.ndarray,
    query_steps: List[int],
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    # BEFORE
    obs0, r0 = replay_episode(theta0, init_state, seed, actions)
    m0 = compute_no_gt_metrics_on_queries(obs_oracle_q, rew_oracle_q, obs0, r0, query_steps)
    loss0 = float(m0["pos_mse"] + m0["bat_mse"] + 0.1 * m0["rew_mae"])
    best_before = {"loss": loss0, **m0}

    best_theta = dict(theta0)
    best_after = dict(best_before)

    for _ in range(int(n_trials)):
        th = sample_theta(rng, best_theta)
        obs_t, r_t = replay_episode(th, init_state, seed, actions)
        m = compute_no_gt_metrics_on_queries(obs_oracle_q, rew_oracle_q, obs_t, r_t, query_steps)
        loss = float(m["pos_mse"] + m["bat_mse"] + 0.1 * m["rew_mae"])
        cand = {"loss": loss, **m}
        if cand["loss"] < best_after["loss"]:
            best_theta = dict(th)
            best_after = dict(cand)

    return best_theta, best_after, best_before


# =========================
# MAIN (PPO-compatible logging)
# =========================
def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H_TOTAL = 50_000
    # A3 uses chunk-style querying (A4-like): per episode we query first K_EP steps (or until termination)
    K_TOTAL = int(np.ceil(H_TOTAL / 5))  # keep your original budget heuristic

    K_EP = 120
    EP_MAX_STEPS = 400
    CALIB_TRIALS = 300
    SLEEP_S = 0.08

    CHARGE_THRESH = 0.30
    SEED0 = 0

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a3_run_{run_id}.jsonl"

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

    t_global = 0
    k_used_total = 0
    ep = 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "start",
            "log_mode": "train",
            "run_id": run_id,
            "baseline": "A3_greedy_charge_querycalib_chunk",
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

        while t_global < H_TOTAL:
            seed = SEED0 + ep
            reset = oracle.reset(seed=int(seed), config_overrides=None)
            sid = reset["session_id"]
            ws0 = reset["world_state"]

            # sync structural params
            theta["grid_size"] = int(ws0.get("grid_size", theta["grid_size"]))
            theta["use_stay"] = bool(ws0.get("use_stay", theta["use_stay"]))
            grid_size = int(theta["grid_size"])

            # start twin from oracle's exact initial state
            init_state = ws0
            env = make_twin_env(theta, init_state=init_state, seed=int(seed))
            obs_t, _ = env.reset(seed=int(seed))
            obs_t = np.asarray(obs_t, dtype=np.float32)

            # per-episode counters for PPO-style plots
            k_used_ep = 0

            # buffers for chunk calibration
            actions_q: List[int] = []
            query_steps: List[int] = [0]  # include reset obs as query step 0
            obs_oracle_q: List[np.ndarray] = [np.asarray(reset["obs"], dtype=np.float32)]
            rew_oracle_q: List[float] = [0.0]  # dummy for t=0
            obs_twin_traj: List[np.ndarray] = [obs_t.copy()]
            rew_twin_traj: List[float] = []

            # episode oracle budget
            k_ep_budget = min(int(K_EP), max(0, int(K_TOTAL - k_used_total)))
            oracle_active = (k_ep_budget > 0)

            f.write(json.dumps({
                "event": "episode_start",
                "log_mode": "train",
                "run_id": run_id,
                "baseline": "A3",
                "ep": int(ep),
                "seed": int(seed),
                "t_global": int(t_global),
                "k_used_total": int(k_used_total),
                "k_used_ep": int(k_used_ep),
                "theta_est": dict(theta),  # optional (kept for debugging)
            }, ensure_ascii=False) + "\n")

            steps_in_ep = 0
            term = False
            trunc = False

            while (not term) and (not trunc) and steps_in_ep < EP_MAX_STEPS and t_global < H_TOTAL:
                a = a3_action(obs_t, grid_size, act_map, CHARGE_THRESH)

                # step twin
                obs_t, r_t, term, trunc, _ = env.step(int(a))
                obs_t = np.asarray(obs_t, dtype=np.float32)
                obs_twin_traj.append(obs_t.copy())
                rew_twin_traj.append(float(r_t))

                queried_oracle = 0
                ws_step = ws0  # fallback

                # chunk querying: query first k_ep_budget steps (if oracle still active)
                if oracle_active and (k_used_ep < k_ep_budget) and (k_used_total < K_TOTAL):
                    step_o = oracle.step(sid, int(a))
                    queried_oracle = 1
                    k_used_total += 1
                    k_used_ep += 1

                    ws_step = step_o.get("world_state", ws0)
                    obs_oracle_q.append(np.asarray(step_o["obs"], dtype=np.float32))
                    rew_oracle_q.append(float(step_o.get("reward", 0.0)))
                    actions_q.append(int(a))
                    query_steps.append(steps_in_ep + 1)  # obs index

                    # stop oracle if episode ended on oracle side
                    if step_o.get("terminated") or step_o.get("truncated"):
                        oracle_active = False

                    if SLEEP_S > 0:
                        time.sleep(float(SLEEP_S))

                # eval-only theta error (if world_state contains those params)
                th_err = theta_mean_abs(theta, ws_step)
                th_err_val = float(th_err) if np.isfinite(th_err) else None

                # log step (PPO-like schema)
                f.write(json.dumps({
                    "event": "step",
                    "log_mode": "train",
                    "run_id": run_id,
                    "baseline": "A3",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(t_global),
                    "_x": int(t_global),
                    "t_ep": int(steps_in_ep),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(k_used_ep),
                    "queried_oracle": int(queried_oracle),

                    # keep theta trajectory available for plot 6
                    "theta_est": dict(theta),

                    # PPO-style acc fields
                    "acc_eval": {"acc/theta_mean_abs": th_err_val},
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                t_global += 1

                if oracle_active and (k_used_ep >= k_ep_budget):
                    oracle_active = False

            # oracle_update (PPO-equivalent) at episode end if we have at least 1 queried step
            if len(actions_q) > 0 and len(obs_oracle_q) >= 2:
                obs_oracle_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                rew_oracle_np = np.asarray(rew_oracle_q, dtype=float)

                obs_twin_np = np.stack(obs_twin_traj, axis=0).astype(np.float32)
                rew_twin_np = np.asarray(rew_twin_traj, dtype=float)

                best_theta, m_after, m_before = calibrate_on_chunk(
                    rng=rng,
                    theta0=dict(theta),
                    init_state=init_state,
                    seed=int(seed),
                    actions=actions_q,
                    obs_oracle_q=obs_oracle_np,
                    rew_oracle_q=rew_oracle_np,
                    query_steps=query_steps,
                    n_trials=int(CALIB_TRIALS),
                )

                theta = dict(best_theta)

                # compute theta error AFTER update vs initial ws0 (eval-only)
                th_err_after = theta_mean_abs(theta, ws0)
                th_err_after_val = float(th_err_after) if np.isfinite(th_err_after) else None

                # PPO-style loss decomposition (baseline-appropriate)
                # total = pos + bat + 0.1*rew
                loss_pos_b = float(m_before["pos_mse"])
                loss_bat_b = float(m_before["bat_mse"])
                loss_rew_b = float(0.1 * m_before["rew_mae"])
                loss_tot_b = float(loss_pos_b + loss_bat_b + loss_rew_b)

                loss_pos_a = float(m_after["pos_mse"])
                loss_bat_a = float(m_after["bat_mse"])
                loss_rew_a = float(0.1 * m_after["rew_mae"])
                loss_tot_a = float(loss_pos_a + loss_bat_a + loss_rew_a)

                f.write(json.dumps({
                    "event": "oracle_update",
                    "log_mode": "train",
                    "run_id": run_id,
                    "baseline": "A3",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(t_global - 1),
                    "_x": int(t_global - 1),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(k_used_ep),
                    "k_ep_budget": int(k_ep_budget),

                    # PPO-style theta columns
                    "theta_est": dict(theta),
                    "theta_true_eval_only": {
                        "battery_max": ws0.get("battery_max", None),
                        "step_cost": ws0.get("step_cost", None),
                        "delivery_reward": ws0.get("delivery_reward", None),
                        "battery_fail_penalty": ws0.get("battery_fail_penalty", None),
                        "p_slip": ws0.get("p_slip", None),
                    },

                    # no-GT operational metrics
                    "acc_no_gt": {
                        "acc/pos_mse": float(m_after["pos_mse"]),
                        "acc/bat_mse": float(m_after["bat_mse"]),
                        "acc/rew_mae": float(m_after["rew_mae"]),
                        "acc/theta_mean_abs": th_err_after_val,  # keep consistent naming
                    },

                    # eval metric preferred
                    "acc_eval": {"acc/theta_mean_abs": th_err_after_val},

                    # loss decomposition (PPO-compatible keys)
                    "loss": {
                        "loss/pos": loss_pos_a,
                        "loss/bat": loss_bat_a,
                        "loss/rew": loss_rew_a,
                        "loss/total": loss_tot_a,
                    },

                    # before/after (useful for debugging)
                    "dbg": {
                        "before": {"loss": loss_tot_b, **m_before},
                        "after": {"loss": loss_tot_a, **m_after},
                    },

                    # if you want a constant "query_cost" for plot 9
                    "query_cost": 1.0,
                }, ensure_ascii=False) + "\n")

            else:
                # PPO-style: update skipped
                f.write(json.dumps({
                    "event": "oracle_update_skipped",
                    "log_mode": "train",
                    "run_id": run_id,
                    "baseline": "A3",
                    "ep": int(ep),
                    "seed": int(seed),
                    "t_global": int(t_global - 1),
                    "_x": int(t_global - 1),
                    "k_used_total": int(k_used_total),
                    "k_used_ep": int(k_used_ep),
                    "reason": "no_oracle_queries_in_episode",
                    "theta_est": dict(theta),
                    "query_cost": 1.0,
                }, ensure_ascii=False) + "\n")

            ep += 1

        f.write(json.dumps({
            "event": "end",
            "log_mode": "train",
            "run_id": run_id,
            "total_twin_steps": int(t_global),
            "k_used_total": int(k_used_total),
            "final_theta": dict(theta),
            "episodes": int(ep),
        }, ensure_ascii=False) + "\n")

    print(f"WROTE: {log_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import requests

from env import RCConfig, make_env
from rc_calib.wrappers import SlipActionWrapper


# Oracle client (HTTP)
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


#  coordinates + action map
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


# Risk-aware charging detour
def a5_policy_action_grid(
    obs: np.ndarray,
    grid_size: int,
    act_map: Dict[str, int],
    step_cost: float,
    safety_steps: int,
    safety_mult: float,
) -> int:
    """
    A5 heuristic:
    - Choose primary target: P if not carrying, else D.
    - Estimate required battery for reaching target (in normalized battery units):
        need ~= (dist * step_cost) * safety_mult
      plus extra buffer for safety_steps.
    - If current battery is too low to safely reach target -> detour to charger C.
    - Else go to primary target.

    Notes:
    - obs[7] is normalized battery in [0,1] (based on battery_max in env).
    - step_cost is a reward cost parameter, not "battery drain". But in this environment
      battery dynamics correlate with steps; we use it as a proxy signal.
    - safety_steps controls how conservative the baseline is.
    """
    has = float(obs[6]) > 0.5
    bat = float(obs[7])  # normalized [0,1]

    rx, ry = obs_to_robot_xy(obs, grid_size)
    px, py = norm_xy_to_grid(float(obs[2]), float(obs[3]), grid_size)
    dx, dy = norm_xy_to_grid(float(obs[4]), float(obs[5]), grid_size)
    cx, cy = norm_xy_to_grid(float(obs[8]), float(obs[9]), grid_size)

    curr = (rx, ry)
    tgt = (dx, dy) if has else (px, py)
    chg = (cx, cy)

    dist_to_tgt = manhattan(curr, tgt)
    dist_to_chg = manhattan(curr, chg)

    # Proxy "needed battery" 
    
    # conservative on purpose
    need = float(dist_to_tgt + safety_steps) * float(step_cost) * float(safety_mult)

    # If battery low relative to estimated need 
    if bat <= need:
        return greedy_action_towards(curr, chg, act_map)
    return greedy_action_towards(curr, tgt, act_map)


# Twin builder
def make_twin_env(theta: Dict[str, Any], seed: int) -> Any:
    cfg = RCConfig(
        grid_size=int(theta["grid_size"]),
        battery_max=int(theta["battery_max"]),
        step_cost=float(theta["step_cost"]),
        delivery_reward=float(theta["delivery_reward"]),
        battery_fail_penalty=float(theta["battery_fail_penalty"]),
        use_stay=bool(theta["use_stay"]),
        seed=seed,
    )
    base = make_env(cfg)
    return SlipActionWrapper(base, p_slip=float(theta["p_slip"]), seed=seed)


# ============================================================
# Simulate policy on twin (for BEFORE/AFTER plots + calibration eval)
# ============================================================

def run_policy_in_twin(
    theta: Dict[str, Any],
    seed: int,
    act_map: Dict[str, int],
    max_steps: int,
    safety_steps: int,
    safety_mult: float,
) -> Tuple[np.ndarray, np.ndarray]:
    env = make_twin_env(theta, seed=seed)
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)

    grid_size = int(theta["grid_size"])
    step_cost = float(theta["step_cost"])

    obs_list: List[np.ndarray] = [obs.copy()]
    r_list: List[float] = []

    term = False
    trunc = False
    for _ in range(max_steps):
        a = a5_policy_action_grid(obs, grid_size, act_map, step_cost, safety_steps, safety_mult)
        obs, r, term, trunc, _ = env.step(int(a))
        obs = np.asarray(obs, dtype=np.float32)
        obs_list.append(obs.copy())
        r_list.append(float(r))
        if term or trunc:
            break

    return np.stack(obs_list, axis=0).astype(np.float32), np.asarray(r_list, dtype=float)


# Metrics
def compute_metrics(obs_p: np.ndarray, r_p: np.ndarray, obs_t: np.ndarray, r_t: np.ndarray) -> Dict[str, float]:
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))
    pos_mse = float(np.mean((obs_p[:T, 0:2] - obs_t[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_p[:Tr]) - np.sum(r_t[:Tr])))
    loss = pos_mse + bat_mse + 0.1 * rew_gap
    return {"loss": float(loss), "pos_mse": pos_mse, "bat_mse": bat_mse, "rew_gap": rew_gap}

def theta_mean_abs(theta_est: Dict[str, Any], theta_true: Dict[str, Any], keys=("battery_max", "step_cost", "p_slip")) -> float:
    diffs = []
    for k in keys:
        if k in theta_est and k in theta_true:
            try:
                diffs.append(abs(float(theta_est[k]) - float(theta_true[k])))
            except Exception:
                pass
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))

# Plotting
def compress_to_arrows(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, float, float]]:
    arrows: List[Tuple[float, float, float, float]] = []
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


def _plot_arrows(ax, arrows: List[Tuple[float, float, float, float]], color: str, label: str):
    if not arrows:
        return
    x0 = np.array([a[0] for a in arrows], dtype=float)
    y0 = np.array([a[1] for a in arrows], dtype=float)
    dx = np.array([a[2] for a in arrows], dtype=float)
    dy = np.array([a[3] for a in arrows], dtype=float)
    ax.quiver(x0, y0, dx, dy, angles="xy", scale_units="xy", scale=1.0, color=color, label=label)


def plot_episode_report(
    title: str,
    obs_oracle_q: np.ndarray,
    r_oracle_q: np.ndarray,
    obs_twin: np.ndarray,
    r_twin: np.ndarray,
    query_steps_in_ep: List[int],
    out_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 4.5))

    # Trajectory
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Trajectory")

    if len(obs_oracle_q) >= 2:
        xq, yq = obs_oracle_q[:, 0], obs_oracle_q[:, 1]
        _plot_arrows(ax1, compress_to_arrows(xq, yq), color="green", label="oracle (queried)")

    xt, yt = obs_twin[:, 0], obs_twin[:, 1]
    _plot_arrows(ax1, compress_to_arrows(xt, yt), color="blue", label="twin")

    ax1.scatter(obs_twin[0, 2], obs_twin[0, 3], label="pickup", s=60)
    ax1.scatter(obs_twin[0, 4], obs_twin[0, 5], label="delivery", s=60)
    ax1.scatter(obs_twin[0, 8], obs_twin[0, 9], label="charger", s=60)

    ax1.scatter(xt[0], yt[0], label="start", s=60)
    ax1.scatter(xt[len(xt) - 1], yt[len(yt) - 1], label="end", s=60)

    for qs in query_steps_in_ep:
        if 0 <= qs < len(obs_twin):
            ax1.scatter(xt[qs], yt[qs], s=20)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="best")

    # Battery
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Battery(t)")
    ax2.plot(obs_twin[:, 7], label="twin")
    if len(obs_oracle_q) >= 2:
        ax2.plot(np.arange(len(obs_oracle_q)), obs_oracle_q[:, 7], label="oracle (queried)")
    for qs in query_steps_in_ep:
        ax2.axvline(qs, linewidth=0.8, alpha=0.35)
    ax2.set_xlabel("t")
    ax2.set_ylabel("battery (normalized)")
    ax2.legend(loc="best")

    # Cum reward
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Cumulative Reward(t)")
    ax3.plot(np.cumsum(r_twin), label="twin")
    if len(r_oracle_q) > 0:
        ax3.plot(np.arange(len(r_oracle_q)), np.cumsum(r_oracle_q), label="oracle (queried)")
    for qs in query_steps_in_ep:
        ax3.axvline(qs, linewidth=0.8, alpha=0.35)
    ax3.set_xlabel("t")
    ax3.set_ylabel("sum reward")
    ax3.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calib_trace(trace: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not trace:
        return

    best_loss = [float(r["best_loss"]) for r in trace]
    best_battery_max = [float(r["best_theta"]["battery_max"]) for r in trace]
    best_step_cost = [float(r["best_theta"]["step_cost"]) for r in trace]
    best_delivery_reward = [float(r["best_theta"]["delivery_reward"]) for r in trace]
    best_battery_fail_penalty = [float(r["best_theta"]["battery_fail_penalty"]) for r in trace]
    best_p_slip = [float(r["best_theta"]["p_slip"]) for r in trace]

    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(3, 2, 1)
    ax.plot(best_loss)
    ax.set_title("best loss")
    ax.set_xlabel("update")

    ax = fig.add_subplot(3, 2, 2)
    ax.plot(best_battery_max)
    ax.set_title("best battery_max")
    ax.set_xlabel("update")

    ax = fig.add_subplot(3, 2, 3)
    ax.plot(best_step_cost)
    ax.set_title("best step_cost")
    ax.set_xlabel("update")

    ax = fig.add_subplot(3, 2, 4)
    ax.plot(best_delivery_reward)
    ax.set_title("best delivery_reward")
    ax.set_xlabel("update")

    ax = fig.add_subplot(3, 2, 5)
    ax.plot(best_battery_fail_penalty)
    ax.set_title("best battery_fail_penalty")
    ax.set_xlabel("update")

    ax = fig.add_subplot(3, 2, 6)
    ax.plot(best_p_slip)
    ax.set_title("best p_slip")
    ax.set_xlabel("update")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


#random search around current theta

def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 400))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.50))
    # keep reward params fixed
    th["delivery_reward"] = float(center["delivery_reward"])
    th["battery_fail_penalty"] = float(center["battery_fail_penalty"])
    th["p_slip"] = float(np.clip(float(center["p_slip"]) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.30))
    return th



def calibrate_on_chunk_policydriven(
    rng: np.random.Generator,
    base_theta: Dict[str, Any],
    seed: int,
    act_map: Dict[str, int],
    obs_oracle: np.ndarray,
    r_oracle: np.ndarray,
    max_steps: int,
    n_trials: int,
    safety_steps: int,
    safety_mult: float,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    best_theta = dict(base_theta)
    obs0, r0 = run_policy_in_twin(best_theta, seed, act_map, max_steps, safety_steps, safety_mult)
    best_m = compute_metrics(obs_oracle, r_oracle, obs0, r0)

    for _ in range(n_trials):
        th = sample_theta(rng, best_theta)
        obs_t, r_t = run_policy_in_twin(th, seed, act_map, max_steps, safety_steps, safety_mult)
        m = compute_metrics(obs_oracle, r_oracle, obs_t, r_t)
        if m["loss"] < best_m["loss"]:
            best_theta, best_m = dict(th), dict(m)

    return best_theta, best_m

def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H_TOTAL = 50000
    K_TOTAL = int(np.ceil(H_TOTAL / 5))  
    SLEEP_S = 0.08

    EP_MAX_STEPS = 400
    QUERY_EVERY = 25
    CALIB_TRIALS = 300

    SAFETY_STEPS = 6     # extra buffer in steps
    SAFETY_MULT = 1.4    # conservative multiplier

    SEED0 = 0

    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    figs_dir = Path("figures"); figs_dir.mkdir(exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a5_run_{run_id}.jsonl"
    trace_path = figs_dir / f"a5_calibtrace_{run_id}.png"

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
    total_oracle_steps = 0
    ep = 0

    calib_trace: List[Dict[str, Any]] = []

    print(f"\nA5 run: {run_id}")
    print(
        f"H={H_TOTAL} | K={K_TOTAL} | sleep={SLEEP_S} | query_every={QUERY_EVERY} "
        f"| safety_steps={SAFETY_STEPS} | safety_mult={SAFETY_MULT}\n"
    )

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "start",
            "run_id": run_id,
            "baseline": "A5_riskaware_charge_querycalib_A4style",
            "oracle_url": ORACLE_URL,
            "H_TOTAL": H_TOTAL,
            "K_TOTAL": K_TOTAL,
            "EP_MAX_STEPS": EP_MAX_STEPS,
            "QUERY_EVERY": QUERY_EVERY,
            "CALIB_TRIALS": CALIB_TRIALS,
            "SAFETY_STEPS": SAFETY_STEPS,
            "SAFETY_MULT": SAFETY_MULT,
            "SLEEP_S": SLEEP_S,
            "init_theta": theta,
        }, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep
            theta_before = dict(theta)

            # oracle reset
            reset = oracle.reset(seed=seed, config_overrides=None)
            sid = reset["session_id"]
            ws0 = reset["world_state"]
            theta_true: Dict[str, Any] = {}
            for k in ("battery_max", "step_cost", "p_slip", "grid_size", "use_stay"):
                if k in ws0 and ws0[k] is not None:
                    theta_true[k] = ws0[k]

            f.write(json.dumps({
                "event": "episode_start",
                "run_id": run_id,
                "ep": ep,
                "seed": seed,
                "t_global": total_twin_steps,
                "theta_true": theta_true,
                "theta_before": dict(theta_before),
                "K_TOTAL": int(K_TOTAL),
                "H_TOTAL": int(H_TOTAL),
                "QUERY_EVERY": int(QUERY_EVERY),
            }, ensure_ascii=False) + "\n")

            grid_size = int(ws0.get("grid_size", theta_before["grid_size"]))
            use_stay = bool(ws0.get("use_stay", theta_before["use_stay"]))

            theta_before["grid_size"] = grid_size
            theta_before["use_stay"] = use_stay

            theta["grid_size"] = grid_size
            theta["use_stay"] = use_stay

            # online twin episode
            env_online = make_twin_env(theta_before, seed=seed)
            obs_t, _ = env_online.reset(seed=seed)
            obs_t = np.asarray(obs_t, dtype=np.float32)

            obs_oracle_q: List[np.ndarray] = [np.asarray(reset["obs"], dtype=np.float32)]
            r_oracle_q: List[float] = []
            query_steps_in_ep: List[int] = []

            steps_in_ep = 0
            term = False
            trunc = False

            while (not term) and (not trunc) and (steps_in_ep < EP_MAX_STEPS) and (total_twin_steps < H_TOTAL):
                a = a5_policy_action_grid(
                    obs=obs_t,
                    grid_size=grid_size,
                    act_map=act_map,
                    step_cost=float(theta_before["step_cost"]),
                    safety_steps=SAFETY_STEPS,
                    safety_mult=SAFETY_MULT,
                )

                # step twin
                obs_t, r_t, term, trunc, _ = env_online.step(int(a))
                obs_t = np.asarray(obs_t, dtype=np.float32)

                queried_now = False

                # query schedule
                if (total_oracle_steps < K_TOTAL) and (steps_in_ep % QUERY_EVERY == 0):
                    step = oracle.step(sid, int(a))
                    total_oracle_steps += 1
                    obs_oracle_q.append(np.asarray(step["obs"], dtype=np.float32))
                    r_oracle_q.append(float(step["reward"]))
                    query_steps_in_ep.append(steps_in_ep)
                    queried_now = True

                    if SLEEP_S > 0:
                        time.sleep(SLEEP_S)

                f.write(json.dumps({
                    "event": "step",
                    "run_id": run_id,
                    "ep": ep,
                    "seed": seed,
                    "t_global": total_twin_steps,
                    "t_ep": steps_in_ep,
                    "queried_oracle": int(queried_now),
                    "action": int(a),
                    "theta": dict(theta_before),
                    "obs_twin": obs_t.tolist(),
                    "reward_twin": float(r_t),
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                total_twin_steps += 1

            # oracle arrays
            if len(obs_oracle_q) >= 2:
                obs_oracle_q_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                r_oracle_q_np = np.asarray(r_oracle_q, dtype=float)
            else:
                obs_oracle_q_np = np.zeros((0, 10), dtype=np.float32)
                r_oracle_q_np = np.zeros((0,), dtype=float)

            # calibration update
            theta_after = dict(theta_before)
            if len(obs_oracle_q_np) >= 2:
                # BEFORE
                obs_b, r_b = run_policy_in_twin(theta_before, seed, act_map, steps_in_ep, SAFETY_STEPS, SAFETY_MULT)
                m_before = compute_metrics(obs_oracle_q_np, r_oracle_q_np, obs_b, r_b)

                best_theta, best_m = calibrate_on_chunk_policydriven(
                    rng=rng,
                    base_theta=dict(theta_before),
                    seed=seed,
                    act_map=act_map,
                    obs_oracle=obs_oracle_q_np,
                    r_oracle=r_oracle_q_np,
                    max_steps=steps_in_ep,
                    n_trials=CALIB_TRIALS,
                    safety_steps=SAFETY_STEPS,
                    safety_mult=SAFETY_MULT,
                )
                theta_after = dict(best_theta)
                theta = dict(theta_after)
                 # comparable theta accuracy
                th_err = theta_mean_abs(theta_after, theta_true, keys=("battery_max", "step_cost", "p_slip"))


                calib_trace.append({
                    "ep": ep,
                    "seed": seed,
                    "best_loss": float(best_m["loss"]),
                    "best_theta": dict(theta_after),
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(best_m),
                    "steps_in_ep": int(steps_in_ep),
                })

                f.write(json.dumps({
                    "event": "calib_update",
                    "run_id": run_id,
                    "ep": ep,
                    "seed": seed,
                    "t_global": total_twin_steps,
                    "k_used_total": int(total_oracle_steps),
                    "theta_true": theta_true,
                    "theta_before": dict(theta_before),
                    "theta_after": dict(theta_after),
                    "acc": {"acc/theta_mean_abs": float(th_err)},
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(best_m),
                }, ensure_ascii=False) + "\n")


            # per-episode plots
            obs_before, r_before = run_policy_in_twin(theta_before, seed, act_map, steps_in_ep, SAFETY_STEPS, SAFETY_MULT)
            obs_after, r_after = run_policy_in_twin(theta_after, seed, act_map, steps_in_ep, SAFETY_STEPS, SAFETY_MULT)

            fig_before = figs_dir / f"a5_before_{run_id}_ep{ep}_seed{seed}.png"
            fig_after = figs_dir / f"a5_after_{run_id}_ep{ep}_seed{seed}.png"

            plot_episode_report(
                title=f"A5 BEFORE (run={run_id}, ep={ep}, seed={seed})",
                obs_oracle_q=obs_oracle_q_np,
                r_oracle_q=r_oracle_q_np,
                obs_twin=obs_before,
                r_twin=r_before,
                query_steps_in_ep=query_steps_in_ep,
                out_path=fig_before,
            )
            plot_episode_report(
                title=f"A5 AFTER (run={run_id}, ep={ep}, seed={seed})",
                obs_oracle_q=obs_oracle_q_np,
                r_oracle_q=r_oracle_q_np,
                obs_twin=obs_after,
                r_twin=r_after,
                query_steps_in_ep=query_steps_in_ep,
                out_path=fig_after,
            )

            ep += 1

        f.write(json.dumps({
            "event": "end",
            "run_id": run_id,
            "total_twin_steps": total_twin_steps,
            "total_oracle_steps": total_oracle_steps,
            "final_theta": theta,
            "calib_updates": len(calib_trace),
        }, ensure_ascii=False) + "\n")

    plot_calib_trace(calib_trace, trace_path)

    print("\nWROTE:")
    print(" -", log_path)
    print(" - per-episode figures: figures/a5_before_*.png and figures/a5_after_*.png")
    print(" - calib trace:", trace_path)
    print("\nNote: 'queried_oracle' markers show when baseline called prototype under query_every schedule.")
    print("Total twin steps:", total_twin_steps, "| Total oracle /step calls:", total_oracle_steps)


if __name__ == "__main__":
    main()


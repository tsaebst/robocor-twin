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


# ============================================================
# Oracle client (HTTP)
# ============================================================

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


# ============================================================
# Helpers: coords + action mapping
# ============================================================

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


# ============================================================
# Twin builder (NO FixedInitWrapper)
# ============================================================

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
    twin = SlipActionWrapper(base, p_slip=float(theta["p_slip"]), seed=seed)
    return twin


# ============================================================
# A4 policy: epsilon-greedy battery-aware
# ============================================================

def a4_action_grid(
    obs: np.ndarray,
    grid_size: int,
    act_map: Dict[str, int],
    charge_thresh: float,
    eps: float,
    rng: np.random.Generator,
    action_space_n: int,
) -> int:
    """
    With prob eps: random action (exploration)
    else: greedy like A3 (go C if battery low else P/D)
    """
    if float(rng.random()) < float(eps):
        return int(rng.integers(0, action_space_n))

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


# ============================================================
# Replay + metrics
# ============================================================

def replay_local(env_obj: Any, actions: List[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    obs, _ = env_obj.reset(seed=seed)
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
    pos_mse = float(np.mean((obs_p[:T, 0:2] - obs_t[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_p[:Tr]) - np.sum(r_t[:Tr])))
    loss = pos_mse + bat_mse + 0.1 * rew_gap
    return {"loss": float(loss), "pos_mse": pos_mse, "bat_mse": bat_mse, "rew_gap": rew_gap}


# ============================================================
# Plotting (same style as A3)
# ============================================================

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


def plot_arrows(ax, arrows: List[Tuple[float, float, float, float]], color: str, label: str):
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
    obs_twin_full: np.ndarray,
    r_twin_full: np.ndarray,
    query_steps_in_ep: List[int],
    out_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 4.5))

    # Trajectory
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Trajectory")

    xq, yq = obs_oracle_q[:, 0], obs_oracle_q[:, 1]
    xt, yt = obs_twin_full[:, 0], obs_twin_full[:, 1]

    plot_arrows(ax1, compress_to_arrows(xq, yq), color="green", label="oracle (queried)")
    plot_arrows(ax1, compress_to_arrows(xt, yt), color="blue", label="twin (full)")

    ax1.scatter(obs_twin_full[0, 2], obs_twin_full[0, 3], label="pickup", s=60)
    ax1.scatter(obs_twin_full[0, 4], obs_twin_full[0, 5], label="delivery", s=60)
    ax1.scatter(obs_twin_full[0, 8], obs_twin_full[0, 9], label="charger", s=60)

    ax1.scatter(xt[0], yt[0], label="start", s=60)
    ax1.scatter(xt[-1], yt[-1], label="end", s=60)

    for qs in query_steps_in_ep:
        if 0 <= qs < len(xt):
            ax1.scatter(xt[qs], yt[qs], s=18)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="best")

    # Battery
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Battery(t)")
    ax2.plot(obs_twin_full[:, 7], label="twin")
    ax2.plot(np.arange(len(obs_oracle_q)), obs_oracle_q[:, 7], label="oracle (queried)")
    for qs in query_steps_in_ep:
        ax2.axvline(qs, linewidth=0.7, alpha=0.35)
    ax2.set_xlabel("t")
    ax2.set_ylabel("battery (normalized)")
    ax2.legend(loc="best")

    # Cum reward
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Cumulative Reward(t)")
    ax3.plot(np.cumsum(r_twin_full), label="twin")
    ax3.plot(np.arange(len(r_oracle_q)), np.cumsum(r_oracle_q), label="oracle (queried)")
    for qs in query_steps_in_ep:
        ax3.axvline(qs, linewidth=0.7, alpha=0.35)
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
    ax.plot(best_loss); ax.set_title("best loss"); ax.set_xlabel("update"); ax.set_ylabel("loss")

    ax = fig.add_subplot(3, 2, 2)
    ax.plot(best_battery_max); ax.set_title("best battery_max"); ax.set_xlabel("update"); ax.set_ylabel("battery_max")

    ax = fig.add_subplot(3, 2, 3)
    ax.plot(best_step_cost); ax.set_title("best step_cost"); ax.set_xlabel("update"); ax.set_ylabel("step_cost")

    ax = fig.add_subplot(3, 2, 4)
    ax.plot(best_delivery_reward); ax.set_title("best delivery_reward"); ax.set_xlabel("update"); ax.set_ylabel("delivery_reward")

    ax = fig.add_subplot(3, 2, 5)
    ax.plot(best_battery_fail_penalty); ax.set_title("best battery_fail_penalty"); ax.set_xlabel("update"); ax.set_ylabel("battery_fail_penalty")

    ax = fig.add_subplot(3, 2, 6)
    ax.plot(best_p_slip); ax.set_title("best p_slip"); ax.set_xlabel("update"); ax.set_ylabel("p_slip")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================
# Calibration (offline random search around current theta)
# ============================================================

def sample_theta(rng: np.random.Generator, center: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(center)
    th["battery_max"] = int(np.clip(int(center["battery_max"]) + int(rng.integers(-80, 81)), 30, 400))
    th["step_cost"] = float(np.clip(float(center["step_cost"]) + float(rng.uniform(-0.10, 0.10)), 0.01, 0.50))
    th["delivery_reward"] = float(np.clip(float(center["delivery_reward"]) + float(rng.uniform(-6.0, 6.0)), 1.0, 20.0))
    th["battery_fail_penalty"] = float(np.clip(float(center["battery_fail_penalty"]) + float(rng.uniform(-8.0, 8.0)), 0.0, 25.0))
    th["p_slip"] = float(np.clip(float(center["p_slip"]) + float(rng.uniform(-0.10, 0.10)), 0.0, 0.30))
    return th


def calibrate_on_chunk(
    rng: np.random.Generator,
    base_theta: Dict[str, Any],
    seed: int,
    actions: List[int],
    obs_oracle: np.ndarray,
    r_oracle: np.ndarray,
    n_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    best_theta = dict(base_theta)
    twin0 = make_twin_env(best_theta, seed=seed)
    obs0, r0 = replay_local(twin0, actions, seed=seed)
    best_m = compute_metrics(obs_oracle, r_oracle, obs0, r0)

    for _ in range(n_trials):
        th = sample_theta(rng, best_theta)
        tw = make_twin_env(th, seed=seed)
        obs_t, r_t = replay_local(tw, actions, seed=seed)
        m = compute_metrics(obs_oracle, r_oracle, obs_t, r_t)
        if m["loss"] < best_m["loss"]:
            best_theta, best_m = dict(th), dict(m)

    return best_theta, best_m


# ============================================================
# Main
# ============================================================

def main():
    ORACLE_URL = "http://16.16.126.90:8001"

    H_TOTAL = 2500
    K_TOTAL = int(np.ceil(H_TOTAL / 5))

    SLEEP_S = 0.08
    K_EP = 120
    EP_MAX_STEPS = 1200
    CALIB_TRIALS = 300

    CHARGE_THRESH = 0.30
    EPS = 0.12  # A4 main knob: exploration rate

    SEED0 = 0

    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    figs_dir = Path("figures"); figs_dir.mkdir(exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a4_run_{run_id}.jsonl"
    trace_path = figs_dir / f"a4_calibtrace_{run_id}.png"

    oracle = RoboCourierOracle(ORACLE_URL)
    act_map = infer_action_mapping_local(seed=SEED0)

    rng = np.random.default_rng(0)

    # Initial theta
    theta = {
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

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "start",
            "run_id": run_id,
            "baseline": "A4_epsgreedy_chunks",
            "oracle_url": ORACLE_URL,
            "H_TOTAL": H_TOTAL,
            "K_TOTAL": K_TOTAL,
            "K_EP": K_EP,
            "EP_MAX_STEPS": EP_MAX_STEPS,
            "CALIB_TRIALS": CALIB_TRIALS,
            "CHARGE_THRESH": CHARGE_THRESH,
            "EPS": EPS,
            "SLEEP_S": SLEEP_S,
            "init_theta": dict(theta),
        }, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep

            reset = oracle.reset(seed=seed, config_overrides=None)
            sid = reset["session_id"]
            obs_o = np.asarray(reset["obs"], dtype=np.float32)
            ws0 = reset["world_state"]
            grid_size = int(ws0.get("grid_size", theta["grid_size"]))
            theta["grid_size"] = grid_size
            theta["use_stay"] = bool(ws0.get("use_stay", theta["use_stay"]))

            twin = make_twin_env(theta, seed=seed)
            obs_t, _ = twin.reset(seed=seed)
            action_space_n = int(getattr(twin, "action_space", None).n) if getattr(twin, "action_space", None) is not None else 4

            actions_q: List[int] = []
            obs_oracle_q: List[np.ndarray] = [obs_o.copy()]
            r_oracle_q: List[float] = []

            obs_twin_full: List[np.ndarray] = [np.asarray(obs_t, dtype=np.float32)]
            r_twin_full: List[float] = []

            query_steps_in_ep: List[int] = []
            steps_in_ep = 0

            k_ep = min(K_EP, max(0, K_TOTAL - total_oracle_steps))
            oracle_active = (k_ep > 0)

            term = False
            trunc = False

            while (not term) and (not trunc) and (steps_in_ep < EP_MAX_STEPS) and (total_twin_steps < H_TOTAL):
                obs_t_np = np.asarray(obs_t, dtype=np.float32)

                a = a4_action_grid(
                    obs=obs_t_np,
                    grid_size=grid_size,
                    act_map=act_map,
                    charge_thresh=CHARGE_THRESH,
                    eps=EPS,
                    rng=rng,
                    action_space_n=action_space_n,
                )

                obs_t, r_t, term, trunc, _ = twin.step(int(a))
                obs_t = np.asarray(obs_t, dtype=np.float32)

                r_twin_full.append(float(r_t))
                obs_twin_full.append(obs_t.copy())

                if oracle_active and (len(actions_q) < k_ep):
                    step = oracle.step(sid, int(a))
                    total_oracle_steps += 1

                    obs_o = np.asarray(step["obs"], dtype=np.float32)
                    obs_oracle_q.append(obs_o.copy())
                    r_oracle_q.append(float(step["reward"]))
                    actions_q.append(int(a))

                    query_steps_in_ep.append(steps_in_ep)

                    if step["terminated"] or step["truncated"]:
                        oracle_active = False

                    if SLEEP_S > 0:
                        time.sleep(SLEEP_S)

                f.write(json.dumps({
                    "event": "step",
                    "run_id": run_id,
                    "ep": ep,
                    "seed": seed,
                    "t_global": total_twin_steps,
                    "t_ep": steps_in_ep,
                    "oracle_queried": int((steps_in_ep in query_steps_in_ep)),
                    "action": int(a),
                    "theta": dict(theta),
                    "eps": float(EPS),
                    "obs_twin": obs_t.tolist(),
                    "reward_twin": float(r_t),
                }, ensure_ascii=False) + "\n")

                steps_in_ep += 1
                total_twin_steps += 1

                if oracle_active and (len(actions_q) >= k_ep):
                    oracle_active = False
                    f.write(json.dumps({
                        "event": "oracle_chunk_end",
                        "run_id": run_id,
                        "ep": ep,
                        "seed": seed,
                        "t_ep": steps_in_ep,
                        "k_ep": k_ep,
                        "total_oracle_steps": total_oracle_steps,
                    }, ensure_ascii=False) + "\n")

            obs_twin_full_np = np.stack(obs_twin_full, axis=0).astype(np.float32)
            r_twin_full_np = np.asarray(r_twin_full, dtype=float)

            if len(actions_q) > 0 and len(obs_oracle_q) >= 2:
                obs_oracle_q_np = np.stack(obs_oracle_q, axis=0).astype(np.float32)
                r_oracle_q_np = np.asarray(r_oracle_q, dtype=float)

                twin_chunk = make_twin_env(theta, seed=seed)
                obs_tc, r_tc = replay_local(twin_chunk, actions_q, seed=seed)
                m_before = compute_metrics(obs_oracle_q_np, r_oracle_q_np, obs_tc, r_tc)

                best_theta, best_m = calibrate_on_chunk(
                    rng=rng,
                    base_theta=dict(theta),
                    seed=seed,
                    actions=actions_q,
                    obs_oracle=obs_oracle_q_np,
                    r_oracle=r_oracle_q_np,
                    n_trials=CALIB_TRIALS,
                )

                theta = dict(best_theta)

                calib_trace.append({
                    "ep": ep,
                    "seed": seed,
                    "best_loss": float(best_m["loss"]),
                    "best_theta": dict(best_theta),
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(best_m),
                    "k_ep": int(k_ep),
                })

                f.write(json.dumps({
                    "event": "calib_update",
                    "run_id": run_id,
                    "ep": ep,
                    "seed": seed,
                    "k_ep": int(k_ep),
                    "metrics_before": dict(m_before),
                    "metrics_after": dict(best_m),
                    "new_theta": dict(best_theta),
                }, ensure_ascii=False) + "\n")

                # per-episode plot
                ep_fig = figs_dir / f"a4_ep{ep}_seed{seed}_{run_id}.png"
                plot_episode_report(
                    title=f"A4 (run={run_id}, ep={ep}, seed={seed})",
                    obs_oracle_q=obs_oracle_q_np,
                    r_oracle_q=r_oracle_q_np,
                    obs_twin_full=obs_twin_full_np,
                    r_twin_full=r_twin_full_np,
                    query_steps_in_ep=query_steps_in_ep,
                    out_path=ep_fig,
                )
            else:
                # twin-only plot
                ep_fig = figs_dir / f"a4_ep{ep}_seed{seed}_{run_id}.png"
                dummy_obs = obs_twin_full_np[:1]
                dummy_r = np.zeros((0,), dtype=float)
                plot_episode_report(
                    title=f"A4 (run={run_id}, ep={ep}, seed={seed}) [twin-only]",
                    obs_oracle_q=dummy_obs,
                    r_oracle_q=dummy_r,
                    obs_twin_full=obs_twin_full_np,
                    r_twin_full=r_twin_full_np,
                    query_steps_in_ep=[],
                    out_path=ep_fig,
                )

            ep += 1

        f.write(json.dumps({
            "event": "end",
            "run_id": run_id,
            "total_twin_steps": total_twin_steps,
            "total_oracle_steps": total_oracle_steps,
            "final_theta": dict(theta),
            "calib_updates": len(calib_trace),
        }, ensure_ascii=False) + "\n")

    plot_calib_trace(calib_trace, trace_path)

    print("\nA4 run:", run_id)
    print("WROTE:")
    print(" -", log_path)
    print(" - per-episode figures: figures/a4_ep*_*.png")
    print(" - calib trace:", trace_path)
    print("Total twin steps:", total_twin_steps, "| Total oracle /step calls:", total_oracle_steps)
    if calib_trace:
        print("Last update best_theta:", calib_trace[-1]["best_theta"])
        print("Last update best_loss:", calib_trace[-1]["best_loss"])


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

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
# Helpers: coordinates + action map
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


# ============================================================
# A3/A5 policy
# ============================================================

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


# ============================================================
# Twin builder
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
    return SlipActionWrapper(base, p_slip=float(theta["p_slip"]), seed=seed)


# ============================================================
# Diagnostics & Success helpers
# ============================================================

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

def compute_success_from_obs(obs_seq: np.ndarray, grid_size: int) -> Dict[str, Any]:
    if len(obs_seq) == 0:
        return {"picked": False, "delivered": False, "success": False, "t_pick": None, "t_deliver": None}
    dx, dy = norm_xy_to_grid(float(obs_seq[0, 4]), float(obs_seq[0, 5]), grid_size)
    has_seq = (obs_seq[:, 6] > 0.5).astype(int)
    t_pick = None
    for t in range(1, len(has_seq)):
        if has_seq[t - 1] == 0 and has_seq[t] == 1:
            t_pick = t
            break
    delivered, t_deliver = False, None
    if t_pick is not None:
        for t in range(t_pick, len(obs_seq)):
            rx, ry = obs_to_robot_xy(obs_seq[t], grid_size)
            if (rx, ry) == (dx, dy):
                delivered = True
                t_deliver = t
                break
    return {"picked": t_pick is not None, "delivered": delivered, "success": bool(t_pick and delivered), "t_pick": t_pick, "t_deliver": t_deliver}

def compute_loop_and_stuck_scores(obs_seq: np.ndarray, grid_size: int, charge_thresh: float) -> Dict[str, Any]:
    T = len(obs_seq)
    if T <= 1: return {"loop_repeat_frac": 0.0, "unique_state_frac": 1.0, "no_progress_max": 0, "no_progress_mean": 0.0}
    seen = set()
    for t in range(T):
        rx, ry = obs_to_robot_xy(obs_seq[t], grid_size)
        has = 1 if float(obs_seq[t, 6]) > 0.5 else 0
        seen.add((rx, ry, has))
    uniq_frac = len(seen) / T
    
    streaks, streak, prev_dist = [], 0, None
    px, py = norm_xy_to_grid(obs_seq[0, 2], obs_seq[0, 3], grid_size)
    dx, dy = norm_xy_to_grid(obs_seq[0, 4], obs_seq[0, 5], grid_size)
    cx, cy = norm_xy_to_grid(obs_seq[0, 8], obs_seq[0, 9], grid_size)
    
    for t in range(T):
        rx, ry = obs_to_robot_xy(obs_seq[t], grid_size)
        has, bat = float(obs_seq[t, 6]) > 0.5, float(obs_seq[t, 7])
        target = (cx, cy) if bat <= charge_thresh else ((dx, dy) if has else (px, py))
        dist = manhattan((rx, ry), target)
        if prev_dist is not None:
            if dist < prev_dist:
                streaks.append(streak); streak = 0
            else: streak += 1
        prev_dist = dist
    streaks.append(streak)
    return {"loop_repeat_frac": 1.0 - uniq_frac, "unique_state_frac": uniq_frac, "no_progress_max": max(streaks) if streaks else 0, "no_progress_mean": np.mean(streaks) if streaks else 0.0}

# ============================================================
# Plotting with Summary Textbox
# ============================================================

def compress_to_arrows(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, float, float]]:
    arrows = []
    if len(x) < 2: return arrows
    dxs, dys = np.diff(x), np.diff(y)
    i = 0
    while i < len(dxs):
        if dxs[i] == 0 and dys[i] == 0: i += 1; continue
        x0, y0, dx_acc, dy_acc, j = x[i], y[i], dxs[i], dys[i], i + 1
        while j < len(dxs) and dxs[j] == dxs[i] and dys[j] == dys[i]:
            dx_acc += dxs[j]; dy_acc += dys[j]; j += 1
        arrows.append((float(x0), float(y0), float(dx_acc), float(dy_acc)))
        i = j
    return arrows

def _plot_arrows(ax, arrows, color, label):
    if not arrows: return
    x0, y0, dx, dy = zip(*arrows)
    ax.quiver(x0, y0, dx, dy, angles="xy", scale_units="xy", scale=1.0, color=color, label=label)

def plot_episode_report(
    title: str,
    obs_oracle_q: np.ndarray,
    r_oracle_q: np.ndarray,
    obs_twin: np.ndarray,
    r_twin: np.ndarray,
    query_steps_in_ep: List[int],
    out_path: Path,
    episode_summary: Optional[Dict[str, Any]] = None
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 4.5))

    # Subplot 1: Trajectory
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Trajectory")
    if len(obs_oracle_q) >= 2:
        _plot_arrows(ax1, compress_to_arrows(obs_oracle_q[:, 0], obs_oracle_q[:, 1]), "green", "oracle (queried)")
    _plot_arrows(ax1, compress_to_arrows(obs_twin[:, 0], obs_twin[:, 1]), "blue", "twin")
    
    ax1.scatter(obs_twin[0, 2], obs_twin[0, 3], label="pickup", s=60)
    ax1.scatter(obs_twin[0, 4], obs_twin[0, 5], label="delivery", s=60)
    ax1.scatter(obs_twin[0, 8], obs_twin[0, 9], label="charger", s=60)
    ax1.legend(loc="best")

    # --- ADD TEXTBOX (A5 Metrics) ---
    if episode_summary:
        txt = (
            f"success={int(episode_summary['success'])} | "
            f"len={episode_summary['episode_length']} | "
            f"Q={episode_summary['oracle_queries']}\n"
            f"loop={episode_summary['loop_repeat_frac']:.2f} | "
            f"no_prog_max={episode_summary['no_progress_max']}"
        )
        ax1.text(
            0.02, 0.02, txt,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    # Battery
    ax2 = fig.add_subplot(1, 3, 2); ax2.set_title("Battery(t)")
    ax2.plot(obs_twin[:, 7], label="twin")
    if len(obs_oracle_q) >= 2: ax2.plot(np.arange(len(obs_oracle_q)), obs_oracle_q[:, 7], label="oracle (queried)")
    for qs in query_steps_in_ep: ax2.axvline(qs, linewidth=0.8, alpha=0.35)
    ax2.legend()

    # Cum reward
    ax3 = fig.add_subplot(1, 3, 3); ax3.set_title("Cumulative Reward(t)")
    ax3.plot(np.cumsum(r_twin), label="twin")
    if len(r_oracle_q) > 0: ax3.plot(np.arange(len(r_oracle_q)), np.cumsum(r_oracle_q), label="oracle (queried)")
    ax3.legend()

    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=160); plt.close(fig)

# ============================================================
# Main Runner
# ============================================================

def main():
    ORACLE_URL = "http://16.16.126.90:8001"
    H_TOTAL, SLEEP_S, CHARGE_THRESH = 2000, 0.08, 0.30
    K_TOTAL, EP_MAX_STEPS, QUERY_EVERY = int(2000/5), 400, 25
    SEED0 = 0

    logs_dir, figs_dir = Path("logs"), Path("figures")
    logs_dir.mkdir(exist_ok=True); figs_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"a5_run_{run_id}.jsonl"

    oracle = RoboCourierOracle(ORACLE_URL)
    act_map = infer_action_mapping_local(seed=SEED0)

    theta = {"grid_size": 10, "use_stay": False, "battery_max": 80, "step_cost": 0.10, "delivery_reward": 10.0, "battery_fail_penalty": 8.0, "p_slip": 0.10}
    total_twin_steps, total_oracle_steps, ep = 0, 0, 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "run_id": run_id, "baseline": "A5", "init_theta": theta}, ensure_ascii=False) + "\n")

        while total_twin_steps < H_TOTAL:
            seed = SEED0 + ep
            # ===== BEFORE CALIBRATION SNAPSHOT (Snapshot Theta) =====
            theta_before = dict(theta)

            reset = oracle.reset(seed=seed); sid = reset["session_id"]; ws0 = reset["world_state"]
            grid_size = int(ws0.get("grid_size", 10))
            theta_before["grid_size"] = grid_size
            
            env_online = make_twin_env(theta_before, seed=seed)
            obs_t, _ = env_online.reset(seed=seed); obs_t = np.asarray(obs_t, dtype=np.float32)

            obs_twin_full, r_twin_full = [obs_t.copy()], []
            obs_oracle_q, r_oracle_q, query_steps_in_ep = [np.asarray(reset["obs"], dtype=np.float32)], [], []
            steps_in_ep, term, trunc = 0, False, False

            while (not term) and (not trunc) and (steps_in_ep < EP_MAX_STEPS) and (total_twin_steps < H_TOTAL):
                a = a3_policy_action_grid(obs_t, grid_size, act_map, CHARGE_THRESH)
                obs_t, r_t, term, trunc, _ = env_online.step(int(a))
                obs_t = np.asarray(obs_t, dtype=np.float32)
                
                obs_twin_full.append(obs_t.copy()); r_twin_full.append(r_t)

                if (total_oracle_steps < K_TOTAL) and (steps_in_ep % QUERY_EVERY == 0):
                    step = oracle.step(sid, int(a))
                    total_oracle_steps += 1
                    obs_oracle_q.append(np.asarray(step["obs"], dtype=np.float32))
                    r_oracle_q.append(float(step["reward"]))
                    query_steps_in_ep.append(steps_in_ep)
                    if SLEEP_S > 0: time.sleep(SLEEP_S)

                steps_in_ep += 1; total_twin_steps += 1

            # --- Episode summary metrics (NEW) ---
            obs_twin_full_np = np.stack(obs_twin_full)
            ep_len = int(len(obs_twin_full_np) - 1)
            oracle_queries = int(len(query_steps_in_ep))
            succ = compute_success_from_obs(obs_twin_full_np, grid_size=grid_size)
            diag = compute_loop_and_stuck_scores(obs_twin_full_np, grid_size=grid_size, charge_thresh=CHARGE_THRESH)

            episode_summary = {
                "ep": int(ep), "seed": int(seed), "episode_length": int(ep_len), "oracle_queries": int(oracle_queries),
                "picked": bool(succ["picked"]), "delivered": bool(succ["delivered"]), "success": bool(succ["success"]),
                "t_pick": succ["t_pick"], "t_deliver": succ["t_deliver"], "loop_repeat_frac": float(diag["loop_repeat_frac"]),
                "unique_state_frac": float(diag["unique_state_frac"]), "no_progress_max": int(diag["no_progress_max"]), "no_progress_mean": float(diag["no_progress_mean"]),
            }

            # Write summary to log
            f.write(json.dumps({"event": "episode_summary", "run_id": run_id, "baseline": "A5", "theta": dict(theta_before), **episode_summary}, ensure_ascii=False) + "\n")

            # Final Plot for this episode (snapshot BEFORE or AFTER depending on logic)
            # Тут ми малюємо фінальний стан епізоду
            plot_episode_report(
                f"A5 Ep {ep} (seed {seed})", 
                np.stack(obs_oracle_q), np.array(r_oracle_q), 
                obs_twin_full_np, np.array(r_twin_full), 
                query_steps_in_ep, 
                figs_dir / f"a5_ep{ep}_{run_id}.png", 
                episode_summary=episode_summary
            )
            ep += 1

if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
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
        payload = {"seed": int(seed), "config_overrides": config_overrides}
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, action: int) -> Dict[str, Any]:
        payload = {"session_id": session_id, "action": int(action)}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


# ============================================================
# Twin fixed init (do not edit env.py)
# ============================================================

def unwrap_env(env_obj: Any, max_depth: int = 15) -> Any:
    """Unwrap .env chains (SlipActionWrapper -> FixedInitWrapper -> RoboCourierEnv)."""
    base = env_obj
    for _ in range(max_depth):
        if hasattr(base, "env"):
            base = getattr(base, "env")
        else:
            break
    return base


def force_world_state_on_env(env_obj: Any, state: Dict[str, Any]) -> None:
    """
    Force RoboCourierEnv to the given internal state.
    Works if env_obj is RoboCourierEnv or a wrapper chain exposing .env.
    """
    base = unwrap_env(env_obj)

    # required attrs
    base.rx, base.ry = int(state["rx"]), int(state["ry"])
    base.px, base.py = int(state["px"]), int(state["py"])
    base.dx, base.dy = int(state["dx"]), int(state["dy"])
    base.cx, base.cy = int(state["cx"]), int(state["cy"])
    base.battery = int(state["battery"])
    base.has_package = bool(state["has_package"])


class FixedInitWrapper:
    """
    Wrapper that forces initial world state right after reset().
    """
    def __init__(self, env_obj: Any, init_state: Dict[str, Any]):
        self.env = env_obj
        self.init_state = init_state
        self.action_space = env_obj.action_space
        self.observation_space = getattr(env_obj, "observation_space", None)

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Force identical initial state
        force_world_state_on_env(self.env, self.init_state)

        # Recompute obs from forced state (robocourier has _obs())
        base = unwrap_env(self.env)
        if hasattr(base, "_obs") and callable(getattr(base, "_obs")):
            obs = base._obs()

        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: Any):
        return self.env.step(action)


def make_twin_env(theta: Dict[str, Any], init_state: Dict[str, Any], seed: int) -> Any:
    """
    Build local twin env with forced init and slip stochasticity.
    """
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
    fixed = FixedInitWrapper(base, init_state=init_state)
    twin = SlipActionWrapper(fixed, p_slip=float(theta["p_slip"]), seed=int(seed))
    return twin


# ============================================================
# Helpers: decode coords + infer action map
# ============================================================

def norm_xy_to_grid(xn: float, yn: float, grid_size: int) -> Tuple[int, int]:
    """
    Convert normalized coords in [0,1] to integer grid coords.
    Use floor-like mapping to reduce 1-cell jitter from rounding.
    """
    gs = grid_size - 1
    x = int(np.floor(float(xn) * gs + 1e-9))
    y = int(np.floor(float(yn) * gs + 1e-9))
    x = max(0, min(gs, x))
    y = max(0, min(gs, y))
    return x, y


def obs_to_robot_xy(obs: np.ndarray, grid_size: int) -> Tuple[int, int]:
    return norm_xy_to_grid(float(obs[0]), float(obs[1]), grid_size)


def infer_action_mapping_local(seed: int = 0) -> Dict[str, int]:
    """
    Infer which discrete action corresponds to U/R/D/L using a local env instance.
    Avoids oracle load.
    """
    e = make_env(RCConfig(seed=int(seed)))
    _obs0, _ = e.reset(seed=int(seed))

    # capture exact internal reset state
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
        if hasattr(e, "_obs"):
            _obs0 = e._obs()

        x0, y0 = obs_to_robot_xy(np.asarray(_obs0), e.grid_size)
        obs1, _, _, _, _ = e.step(int(a))
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

    # optional "stay" if present
    if e.action_space.n == 5:
        # find which action yields no movement from same state
        for a in range(e.action_space.n):
            e.reset(seed=int(seed))
            force_world_state_on_env(e, state0)
            if hasattr(e, "_obs"):
                _obs0 = e._obs()
            x0, y0 = obs_to_robot_xy(np.asarray(_obs0), e.grid_size)
            obs1, _, _, _, _ = e.step(int(a))
            x1, y1 = obs_to_robot_xy(np.asarray(obs1), e.grid_size)
            if (x1 - x0, y1 - y0) == (0, 0):
                mapping["stay"] = a
                break

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

    return act.get("stay", act["up"])


# ============================================================
# Oracle rollout under heuristic A1 (Greedy P->D, ignore charger)
# ============================================================

def oracle_rollout_greedy_A1(
    oracle: RoboCourierOracle,
    seed: int,
    k_steps: int,
    sleep_s: float,
    act_map: Dict[str, int],
) -> Dict[str, Any]:
    """
    Collect up to k oracle /step calls (stop early if terminated/truncated).
    Returns actions, obs, rewards, world_states, init_state.
    """
    reset = oracle.reset(seed=int(seed), config_overrides=None)
    sid = reset["session_id"]
    obs = np.asarray(reset["obs"], dtype=np.float32)
    ws0 = reset["world_state"]

    obs_list: List[List[float]] = [obs.tolist()]
    ws_list: List[Dict[str, Any]] = [ws0]
    r_list: List[float] = []
    a_list: List[int] = []

    grid_size = int(ws0["grid_size"])

    for _ in range(int(k_steps)):
        has = float(obs[6]) > 0.5
        rx, ry = obs_to_robot_xy(obs, grid_size)

        if not has:
            px, py = norm_xy_to_grid(float(obs[2]), float(obs[3]), grid_size)
            target = (px, py)
        else:
            dx, dy = norm_xy_to_grid(float(obs[4]), float(obs[5]), grid_size)
            target = (dx, dy)

        a = greedy_action_towards((rx, ry), target, act_map)

        step = oracle.step(sid, a)
        obs = np.asarray(step["obs"], dtype=np.float32)

        a_list.append(int(a))
        r_list.append(float(step["reward"]))
        obs_list.append(obs.tolist())
        ws_list.append(step["world_state"])

        if bool(step.get("terminated")) or bool(step.get("truncated")):
            break

        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

    return {
        "session_id": sid,
        "seed": int(seed),
        "k_steps": int(k_steps),
        "sleep_s": float(sleep_s),
        "actions": a_list,
        "obs": obs_list,
        "rewards": r_list,
        "world_states": ws_list,
        "init_state": ws0,
    }


# ============================================================
# Replay in twin using fixed actions
# ============================================================

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


# ============================================================
# Metrics
# ============================================================

def find_transition_time(has_seq: np.ndarray, from_val: float, to_val: float) -> Optional[int]:
    for i in range(1, len(has_seq)):
        if has_seq[i - 1] == from_val and has_seq[i] == to_val:
            return i
    return None


def compute_metrics(obs_p: np.ndarray, r_p: np.ndarray, obs_t: np.ndarray, r_t: np.ndarray) -> Dict[str, float]:
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))
    if T <= 1:
        # degenerate
        return {"loss": 1e9, "pos_mse": 1e9, "bat_mse": 1e9, "rew_gap": 1e9, "pick_err": 1e3, "drop_err": 1e3}

    pos_mse = float(np.mean((obs_p[:T, 0:2] - obs_t[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_p[:Tr]) - np.sum(r_t[:Tr])))

    has_p = np.round(obs_p[:T, 6]).astype(float)
    has_t = np.round(obs_t[:T, 6]).astype(float)

    t_pick_p = find_transition_time(has_p, 0.0, 1.0)
    t_pick_t = find_transition_time(has_t, 0.0, 1.0)
    t_drop_p = find_transition_time(has_p, 1.0, 0.0)
    t_drop_t = find_transition_time(has_t, 1.0, 0.0)

    def err(a: Optional[int], b: Optional[int], penalty: float = 1e3) -> float:
        if a is None or b is None:
            return float(penalty)
        return float(abs(a - b))

    pick_err = err(t_pick_p, t_pick_t)
    drop_err = err(t_drop_p, t_drop_t)

    loss = pos_mse + bat_mse + 0.1 * rew_gap + 0.01 * (pick_err + drop_err)

    return {
        "loss": float(loss),
        "pos_mse": pos_mse,
        "bat_mse": bat_mse,
        "rew_gap": rew_gap,
        "pick_err": float(pick_err),
        "drop_err": float(drop_err),
    }


# ============================================================
# Plotting: arrows + landmarks
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


def plot_arrows(arrows: List[Tuple[float, float, float, float]], color: str, label: str):
    if not arrows:
        return

    x0 = np.array([a[0] for a in arrows], dtype=float)
    y0 = np.array([a[1] for a in arrows], dtype=float)
    dx = np.array([a[2] for a in arrows], dtype=float)
    dy = np.array([a[3] for a in arrows], dtype=float)

    plt.quiver(
        x0, y0, dx, dy,
        angles="xy", scale_units="xy", scale=1.0,
        color=color, label=label, width=0.004, headwidth=3.5, headlength=5.0
    )


def plot_report(
    obs_p: np.ndarray,
    r_p: np.ndarray,
    obs_t: np.ndarray,
    r_t: np.ndarray,
    title: str,
    out_prefix: Path,
):
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))

    # 1) trajectory arrows
    plt.figure(figsize=(7, 6))
    x_p, y_p = obs_p[:T, 0], obs_p[:T, 1]
    x_t, y_t = obs_t[:T, 0], obs_t[:T, 1]

    plot_arrows(compress_to_arrows(x_p, y_p), color="green", label="oracle/prototype")
    plot_arrows(compress_to_arrows(x_t, y_t), color="blue", label="twin")

    # start/end
    plt.scatter(x_p[0], y_p[0], label="proto start")
    plt.scatter(x_p[T - 1], y_p[T - 1], label="proto end")
    plt.scatter(x_t[0], y_t[0], label="twin start")
    plt.scatter(x_t[T - 1], y_t[T - 1], label="twin end")

    # landmarks from obs[0] (normalized coords)
    plt.scatter(obs_p[0, 2], obs_p[0, 3], label="pickup (P)")
    plt.scatter(obs_p[0, 4], obs_p[0, 5], label="delivery (D)")
    plt.scatter(obs_p[0, 8], obs_p[0, 9], label="charger (C)")

    plt.title(f"{title} - Trajectory (arrows)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".traj.png"), dpi=160)
    plt.close()

    # 2) battery
    plt.figure(figsize=(7, 4))
    plt.plot(obs_p[:T, 7], label="oracle battery")
    plt.plot(obs_t[:T, 7], "--", label="twin battery")
    plt.title(f"{title} - Battery(t)")
    plt.xlabel("t")
    plt.ylabel("battery (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".bat.png"), dpi=160)
    plt.close()

    # 3) cumulative reward
    plt.figure(figsize=(7, 4))
    plt.plot(np.cumsum(r_p[:Tr]), label="oracle cum reward")
    plt.plot(np.cumsum(r_t[:Tr]), "--", label="twin cum reward")
    plt.title(f"{title} - Cumulative Reward(t)")
    plt.xlabel("t")
    plt.ylabel("sum reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".rew.png"), dpi=160)
    plt.close()


def plot_calib_trace(best_losses: List[float], best_thetas: List[Dict[str, Any]], out_prefix: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(best_losses)
    plt.title("Calibration trace: best loss vs trial")
    plt.xlabel("trial")
    plt.ylabel("best loss")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".bestloss.png"), dpi=160)
    plt.close()

    keys = ["battery_max", "step_cost", "delivery_reward", "battery_fail_penalty", "p_slip"]
    for k in keys:
        plt.figure(figsize=(7, 4))
        plt.plot([float(th[k]) for th in best_thetas])
        plt.title(f"Calibration trace: best {k} vs trial")
        plt.xlabel("trial")
        plt.ylabel(k)
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix(f".best_{k}.png"), dpi=160)
        plt.close()


# ============================================================
# Calibration (random search, offline)
# ============================================================

def sample_theta(rng: np.random.Generator, base: Dict[str, Any]) -> Dict[str, Any]:
    th = dict(base)
    th["battery_max"] = int(rng.integers(30, 401))
    th["step_cost"] = float(rng.uniform(0.01, 0.50))
    th["delivery_reward"] = float(rng.uniform(1.0, 20.0))
    th["battery_fail_penalty"] = float(rng.uniform(0.0, 25.0))
    th["p_slip"] = float(rng.uniform(0.0, 0.30))
    return th


def main():
    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------
    ORACLE_URL = "http://16.16.126.90:8001"
    SEED = 0

    # H controls desired horizon, but oracle only provides K=ceil(H/5) steps
    H = 200
    K = int(np.ceil(H / 5))  # oracle budget
    SLEEP_S = 0.08           # rate limit to protect EC2
    N_TRIALS = 300

    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    figs_dir = Path("figures"); figs_dir.mkdir(exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    log_rollout = logs_dir / f"a1_oracle_rollout_{run_id}.jsonl"
    log_calib = logs_dir / f"a1_calib_trials_{run_id}.jsonl"

    oracle = RoboCourierOracle(ORACLE_URL)

    # Infer action map locally (no oracle load)
    act_map = infer_action_mapping_local(seed=SEED)

    # ------------------------------------------------------------
    # 1) Collect oracle rollout under heuristic A1
    # ------------------------------------------------------------
    roll = oracle_rollout_greedy_A1(
        oracle=oracle,
        seed=SEED,
        k_steps=K,
        sleep_s=SLEEP_S,
        act_map=act_map,
    )

    with log_rollout.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "event": "oracle_rollout", "rollout": roll}, ensure_ascii=False) + "\n")

    actions = roll["actions"]
    init_state = roll["init_state"]
    obs_oracle = np.asarray(roll["obs"], dtype=np.float32)
    r_oracle = np.asarray(roll["rewards"], dtype=float)

    # ------------------------------------------------------------
    # 2) Initial theta + BEFORE plots
    # ------------------------------------------------------------
    base_theta = {
        "grid_size": int(init_state["grid_size"]),
        "use_stay": bool(init_state["use_stay"]),
        "battery_max": int(init_state["battery_max"]),
        "step_cost": float(init_state["step_cost"]),
        "delivery_reward": float(init_state["delivery_reward"]),
        "battery_fail_penalty": float(init_state["battery_fail_penalty"]),
        "p_slip": 0.15,  # intentional mismatch knob
    }

    twin0 = make_twin_env(base_theta, init_state=init_state, seed=SEED)
    obs_t0, r_t0 = replay_local(twin0, actions, seed=SEED)
    m0 = compute_metrics(obs_oracle, r_oracle, obs_t0, r_t0)

    print("\nA1 baseline run:", run_id)
    print("Oracle budget K =", K, "| sleep_s =", SLEEP_S, "| N_TRIALS =", N_TRIALS)
    print("Initial theta:", base_theta)
    print("Initial metrics:", m0)

    plot_report(
        obs_oracle, r_oracle, obs_t0, r_t0,
        title=f"A1 BEFORE (run={run_id})",
        out_prefix=figs_dir / f"a1_before_{run_id}"
    )

    # ------------------------------------------------------------
    # 3) Random search calibration (OFFLINE)
    # ------------------------------------------------------------
    rng = np.random.default_rng(0)

    best_theta = dict(base_theta)
    best_metrics = dict(m0)

    best_losses: List[float] = [float(best_metrics["loss"])]
    best_thetas: List[Dict[str, Any]] = [dict(best_theta)]

    with log_calib.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "event": "start",
            "baseline": "A1_greedy_PD_no_charge",
            "oracle_url": ORACLE_URL,
            "seed": SEED,
            "H": H,
            "K": K,
            "sleep_s": SLEEP_S,
            "init_theta": base_theta,
            "init_metrics": m0,
        }, ensure_ascii=False) + "\n")

        for i in range(N_TRIALS):
            theta = sample_theta(rng, base=base_theta)

            twin = make_twin_env(theta, init_state=init_state, seed=SEED)
            obs_t, r_t = replay_local(twin, actions, seed=SEED)

            m = compute_metrics(obs_oracle, r_oracle, obs_t, r_t)

            rec = {"run_id": run_id, "trial": i, **{k: theta[k] for k in theta.keys()}, **m}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if m["loss"] < best_metrics["loss"]:
                best_theta = dict(theta)
                best_metrics = dict(m)

                theta_view = {
                    "battery_max": best_theta["battery_max"],
                    "step_cost": best_theta["step_cost"],
                    "delivery_reward": best_theta["delivery_reward"],
                    "battery_fail_penalty": best_theta["battery_fail_penalty"],
                    "p_slip": best_theta["p_slip"],
                }

                print(
                    f"[best @ {i}] "
                    f"loss={best_metrics['loss']:.6f} "
                    f"pos_mse={best_metrics['pos_mse']:.6f} "
                    f"bat_mse={best_metrics['bat_mse']:.6f} "
                    f"rew_gap={best_metrics['rew_gap']:.6f} "
                    f"pick_err={best_metrics['pick_err']:.1f} "
                    f"drop_err={best_metrics['drop_err']:.1f} "
                    f"theta={theta_view}"
                )

            best_losses.append(float(best_metrics["loss"]))
            best_thetas.append(dict(best_theta))

        f.write(json.dumps({"run_id": run_id, "event": "end", "best_theta": best_theta, "best_metrics": best_metrics}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------
    # 4) AFTER plots + calibration traces
    # ------------------------------------------------------------
    twin_best = make_twin_env(best_theta, init_state=init_state, seed=SEED)
    obs_tb, r_tb = replay_local(twin_best, actions, seed=SEED)

    plot_report(
        obs_oracle, r_oracle, obs_tb, r_tb,
        title=f"A1 AFTER (run={run_id})",
        out_prefix=figs_dir / f"a1_after_{run_id}"
    )

    plot_calib_trace(best_losses, best_thetas, out_prefix=figs_dir / f"a1_calibtrace_{run_id}")

    print("\nBEST THETA:", best_theta)
    print("BEST METRICS:", best_metrics)
    print("\nWROTE:")
    print(" -", log_rollout)
    print(" -", log_calib)
    print(" - figures/a1_before_{run_id}.* , figures/a1_after_{run_id}.* , figures/a1_calibtrace_{run_id}.*")


if __name__ == "__main__":
    main()

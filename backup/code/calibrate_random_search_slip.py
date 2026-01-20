from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from env import RCConfig, make_env
from rc_calib.wrappers import SlipActionWrapper


# -----------------------------
# Twin-only fixed init (no edits to env.py)
# -----------------------------

def extract_world_state_from_env(env) -> Dict[str, Any]:
    """
    Extracts full 'world state' from RoboCourierEnv instance after reset().
    Works because robocourier's env object stores these as attributes.
    """
    return {
        "rx": int(env.rx), "ry": int(env.ry),
        "px": int(env.px), "py": int(env.py),
        "dx": int(env.dx), "dy": int(env.dy),
        "cx": int(env.cx), "cy": int(env.cy),
        "battery": int(env.battery),
        "has_package": bool(env.has_package),
    }


def force_world_state_on_env(env, state: Dict[str, Any]) -> None:
    """
    Forces RoboCourierEnv instance to a given state by writing its attributes.
    """
    env.rx, env.ry = state["rx"], state["ry"]
    env.px, env.py = state["px"], state["py"]
    env.dx, env.dy = state["dx"], state["dy"]
    env.cx, env.cy = state["cx"], state["cy"]
    env.battery = state["battery"]
    env.has_package = state["has_package"]


class FixedInitTwinWrapper:
    """
    Wraps an env and forces its initial world state right after reset().
    This avoids editing the original environment implementation.
    """
    def __init__(self, env, init_state: Dict[str, Any]):
        self.env = env
        self.init_state = init_state

        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Force identical initial state
        force_world_state_on_env(self.env, self.init_state)

        # Recompute obs from forced state (robocourier has _obs())
        if hasattr(self.env, "_obs") and callable(getattr(self.env, "_obs")):
            obs = self.env._obs()

        return np.array(obs, dtype=np.float32), info

    def step(self, action: Any):
        return self.env.step(action)


# -----------------------------
# Core rollout utilities
# -----------------------------
def obs_to_grid_xy(obs: np.ndarray, grid_size: int) -> Tuple[int, int]:
    """
    obs contains normalized x,y in [0,1].
    Convert to integer grid coords in [0, grid_size-1].
    """
    gs = grid_size - 1
    x = int(round(float(obs[0]) * gs))
    y = int(round(float(obs[1]) * gs))
    x = max(0, min(gs, x))
    y = max(0, min(gs, y))
    return x, y


def infer_action_mapping(env, seed: int = 0) -> Dict[str, int]:
    """
    Infers which discrete action corresponds to moving U/R/D/L
    by probing one-step transitions from the same initial state.
    Assumes deterministic base env (prototype) for inference.
    """
    obs0, _ = env.reset(seed=seed)
    # capture initial state attributes (exact)
    state0 = extract_world_state_from_env(env)

    mapping: Dict[str, int] = {}
    for a in range(env.action_space.n):
        # reset back to identical state
        env.reset(seed=seed)
        force_world_state_on_env(env, state0)
        if hasattr(env, "_obs"):
            obs0 = env._obs()

        x0, y0 = obs_to_grid_xy(np.array(obs0), env.grid_size)
        obs1, _, _, _, _ = env.step(a)
        x1, y1 = obs_to_grid_xy(np.array(obs1), env.grid_size)

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
        raise RuntimeError(f"Could not infer full action mapping, got: {mapping}")
    return mapping

def greedy_step_towards(curr: Tuple[int, int], target: Tuple[int, int], act: Dict[str, int]) -> int:
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
    # already at target: arbitrary (won't be used)
    return act["up"]


def norm_xy_to_grid(xn: float, yn: float, grid_size: int) -> Tuple[int, int]:
    gs = grid_size - 1
    x = int(round(float(xn) * gs))
    y = int(round(float(yn) * gs))
    x = max(0, min(gs, x))
    y = max(0, min(gs, y))
    return x, y


def collect_actions_heuristic(env, n_steps: int, seed: int, force_state: Optional[Dict[str, Any]] = None,
                              low_battery_thresh: float = 0.25) -> List[int]:
    """
    Heuristic: go to Pickup if no package; else go to Delivery.
    If battery low, go to Charger first (simple rule).
    """
    obs, _ = env.reset(seed=seed)
    if force_state is not None:
        force_world_state_on_env(env, force_state)
        if hasattr(env, "_obs"):
            obs = env._obs()

    # infer action mapping once (from this env)
    act = infer_action_mapping(env, seed=seed)

    actions: List[int] = []
    for _ in range(n_steps):
        obs_arr = np.array(obs, dtype=float)

        # decode grid positions from obs
        gs = env.grid_size
        rx, ry = norm_xy_to_grid(obs_arr[0], obs_arr[1], gs)
        px, py = norm_xy_to_grid(obs_arr[2], obs_arr[3], gs)
        dx, dy = norm_xy_to_grid(obs_arr[4], obs_arr[5], gs)
        cx, cy = norm_xy_to_grid(obs_arr[8], obs_arr[9], gs)


        has = float(obs_arr[6]) > 0.5
        bat = float(obs_arr[7])  # normalized battery in [0,1]

        # choose target
        if bat <= low_battery_thresh:
            target = (cx, cy)
        else:
            target = (dx, dy) if has else (px, py)

        a = greedy_step_towards((rx, ry), target, act)
        actions.append(a)

        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break

    return actions


def collect_actions_heuristic(
    env,
    n_steps: int,
    act_map: Dict[str, int],
    seed: int = 0,
    force_state: Optional[Dict[str, Any]] = None,
) -> List[int]:

    obs, info = env.reset(seed=seed)

    if force_state is not None:
        force_world_state_on_env(env, force_state)
        if hasattr(env, "_obs"):
            obs = env._obs()

    actions: List[int] = []

    for _ in range(n_steps):
        obs_arr = np.asarray(obs, dtype=float)

        # robot
        rx, ry = norm_xy_to_grid(obs_arr[0], obs_arr[1], env.grid_size)

        # pickup / delivery
        has = obs_arr[6] > 0.5
        if not has:
            tx, ty = norm_xy_to_grid(obs_arr[2], obs_arr[3], env.grid_size)  # pickup
        else:
            tx, ty = norm_xy_to_grid(obs_arr[4], obs_arr[5], env.grid_size)  # delivery

        # greedy move
        if rx < tx:
            a = act_map["right"]
        elif rx > tx:
            a = act_map["left"]
        elif ry < ty:
            a = act_map["up"]
        elif ry > ty:
            a = act_map["down"]
        else:
            break  # reached target

        actions.append(a)
        obs, r, term, trunc, info = env.step(a)

        if term or trunc:
            break

    return actions



def replay(env, actions: List[Any], seed: int = 0, force_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replay a fixed action trace. Optionally force world state after reset (prototype control).
    """
    obs, info = env.reset(seed=seed)
    if force_state is not None:
        force_world_state_on_env(env, force_state)
        if hasattr(env, "_obs") and callable(getattr(env, "_obs")):
            obs = env._obs()

    obs_list = [np.array(obs, dtype=float)]
    r_list: List[float] = []

    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(np.array(obs, dtype=float))
        r_list.append(float(r))
        if term or trunc:
            break

    return np.stack(obs_list, axis=0), np.array(r_list, dtype=float)


# -----------------------------
# Loss / metrics
# -----------------------------

def compute_loss(
    obs_p: np.ndarray, r_p: np.ndarray,
    obs_t: np.ndarray, r_t: np.ndarray,
    w_pos: float = 1.0,
    w_bat: float = 1.0,
    w_rew: float = 0.1
) -> Dict[str, float]:
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))

    pos_mse = float(np.mean((obs_p[:T, 0:2] - obs_t[:T, 0:2]) ** 2))
    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_p[:Tr]) - np.sum(r_t[:Tr])))

    loss = w_pos * pos_mse + w_bat * bat_mse + w_rew * rew_gap
    return {"loss": float(loss), "pos_mse": pos_mse, "bat_mse": bat_mse, "rew_gap": rew_gap}


# -----------------------------
# Plotting helpers (arrows)
# -----------------------------

def compress_to_arrows(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Convert step-by-step positions into a list of compressed arrows.
    Each arrow aggregates consecutive moves in the same direction.
    Returns list of (x0, y0, dx, dy).
    """
    arrows: List[Tuple[float, float, float, float]] = []
    if len(x) < 2:
        return arrows

    dxs = np.diff(x)
    dys = np.diff(y)

    # Ignore zero-moves (shouldn't happen often, but safe)
    steps = [(dxs[i], dys[i]) for i in range(len(dxs)) if not (dxs[i] == 0 and dys[i] == 0)]
    if not steps:
        return arrows

    # Walk original indices to keep correct start points
    i = 0
    while i < len(dxs):
        if dxs[i] == 0 and dys[i] == 0:
            i += 1
            continue

        x0, y0 = x[i], y[i]
        dx_acc, dy_acc = dxs[i], dys[i]

        j = i + 1
        # same direction = exactly equal delta (grid-normalized, so exact is fine)
        while j < len(dxs) and dxs[j] == dxs[i] and dys[j] == dys[i]:
            dx_acc += dxs[j]
            dy_acc += dys[j]
            j += 1

        arrows.append((float(x0), float(y0), float(dx_acc), float(dy_acc)))
        i = j

    return arrows


def plot_arrows(arrows: List[Tuple[float, float, float, float]], color: str, label: str):
    """
    Draw arrows (x0,y0,dx,dy) using quiver.
    """
    if not arrows:
        return
    x0 = np.array([a[0] for a in arrows], dtype=float)
    y0 = np.array([a[1] for a in arrows], dtype=float)
    dx = np.array([a[2] for a in arrows], dtype=float)
    dy = np.array([a[3] for a in arrows], dtype=float)

    plt.quiver(
        x0, y0, dx, dy,
        angles="xy", scale_units="xy", scale=1.0,
        color=color, width=0.004, headwidth=3.5, headlength=5.0,
        label=label
    )


# -----------------------------
# Plotting
# -----------------------------

def plot_paired(obs_p: np.ndarray, r_p: np.ndarray, obs_t: np.ndarray, r_t: np.ndarray,
                title_prefix: str, out_prefix: Path):
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))

    x_p, y_p = obs_p[:T, 0], obs_p[:T, 1]
    x_t, y_t = obs_t[:T, 0], obs_t[:T, 1]

    # --- 1) Trajectories as ARROWS + key landmarks ---
    plt.figure()

    # compressed arrow segments
    arrows_p = compress_to_arrows(x_p, y_p)
    arrows_t = compress_to_arrows(x_t, y_t)

    # Prototype: green arrows; Twin: blue arrows
    plot_arrows(arrows_p, color="green", label="prototype (arrows)")
    plot_arrows(arrows_t, color="blue", label="twin (arrows)")

    # Start / End markers
    plt.scatter(x_p[0], y_p[0], label="proto start")
    plt.scatter(x_p[T-1], y_p[T-1], label="proto end")
    plt.scatter(x_t[0], y_t[0], label="twin start")
    plt.scatter(x_t[T-1], y_t[T-1], label="twin end")

    # Landmarks: pickup/delivery/charger from obs[0] (constant for episode)
    px, py = obs_p[0, 2], obs_p[0, 3]
    dx, dy = obs_p[0, 4], obs_p[0, 5]
    cx, cy = obs_p[0, 8], obs_p[0, 9]
    plt.scatter(px, py, label="pickup (P)")
    plt.scatter(dx, dy, label="delivery (D)")
    plt.scatter(cx, cy, label="charger (C)")

    # Pickup/Dropoff events from has_package (obs[6])
    has_p = obs_p[:T, 6]
    has_t = obs_t[:T, 6]

    def find_transitions(has_arr, from_val, to_val):
        idx = []
        for i in range(1, len(has_arr)):
            if has_arr[i-1] == from_val and has_arr[i] == to_val:
                idx.append(i)
        return idx

    p_pick = find_transitions(has_p, 0.0, 1.0)
    p_drop = find_transitions(has_p, 1.0, 0.0)
    t_pick = find_transitions(has_t, 0.0, 1.0)
    t_drop = find_transitions(has_t, 1.0, 0.0)

    for i in p_pick:
        plt.scatter(x_p[i], y_p[i], label="proto pickup event")
    for i in p_drop:
        plt.scatter(x_p[i], y_p[i], label="proto dropoff event")
    for i in t_pick:
        plt.scatter(x_t[i], y_t[i], label="twin pickup event")
    for i in t_drop:
        plt.scatter(x_t[i], y_t[i], label="twin dropoff event")

    plt.title(f"{title_prefix} - Paired trajectories (arrows)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".traj.png"), dpi=160)
    plt.close()

    # --- 2) Battery overlay ---
    plt.figure()
    plt.plot(obs_p[:T, 7], label="prototype battery (obs[7])")
    plt.plot(obs_t[:T, 7], "--", label="twin battery (obs[7])")
    plt.title(f"{title_prefix} - Battery over time")
    plt.xlabel("t"); plt.ylabel("battery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".bat.png"), dpi=160)
    plt.close()

    # --- 3) Cumulative reward overlay ---
    plt.figure()
    plt.plot(np.cumsum(r_p[:Tr]), label="prototype cum reward")
    plt.plot(np.cumsum(r_t[:Tr]), "--", label="twin cum reward")
    plt.title(f"{title_prefix} - Cumulative reward")
    plt.xlabel("t"); plt.ylabel("sum reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".rew.png"), dpi=160)
    plt.close()


# -----------------------------
# Random search calibration
# -----------------------------

def sample_theta(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "p_slip": float(rng.uniform(0.0, 0.30)),
        "step_cost": float(rng.uniform(0.01, 0.50)),
        "battery_fail_penalty": float(rng.uniform(0.0, 20.0)),
    }


def make_twin_env(base_cfg: RCConfig, theta: Dict[str, Any], seed: int, init_state: Dict[str, Any]) -> Any:
    """
    Create twin env with:
      1) base robocourier env
      2) forced initial world state (FixedInitTwinWrapper)
      3) stochastic dynamics mismatch (SlipActionWrapper)
    """
    cfg_dict = asdict(base_cfg)
    cfg_dict.update({
        "step_cost": float(theta["step_cost"]),
        "battery_fail_penalty": float(theta["battery_fail_penalty"]),
        "seed": seed,
    })

    base = make_env(RCConfig(**cfg_dict))
    fixed = FixedInitTwinWrapper(base, init_state=init_state)
    twin = SlipActionWrapper(fixed, p_slip=float(theta["p_slip"]), seed=seed)
    return twin


def main():
    # 1. Створюємо базові налаштування та директорії
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    figs_dir = Path("figures")
    figs_dir.mkdir(exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"calib_random_search_slip_{run_id}.jsonl"

    # Параметри експерименту
    n_steps = 200
    seed_actions = 0

    # 2. Ініціалізуємо прототип (Ground Truth)
    proto_cfg = RCConfig(battery_max=200, seed=0)
    proto_env = make_env(proto_cfg)

    # Важливо: reset() виконуємо до того, як витягувати стан або інферити дії
    proto_env.reset(seed=seed_actions)
    init_state = extract_world_state_from_env(proto_env)

    # 3. Інферимо семантику дій (один раз)
    act_map = infer_action_mapping(proto_env, seed=seed_actions)

    # 4. Генеруємо евристичну траєкторію НА ПРОТОТИПІ
    actions = collect_actions_heuristic(
        proto_env, 
        n_steps=n_steps, 
        act_map=act_map, 
        seed=seed_actions, 
        force_state=init_state
    )

    # 5. Replay на прототипі для отримання еталону (obs_p, r_p)
    obs_p, r_p = replay(proto_env, actions, seed=seed_actions, force_state=init_state)

    # 6. Налаштування "двійника" (Twin)
    twin_base_cfg = RCConfig(grid_size=proto_cfg.grid_size, battery_max=proto_cfg.battery_max, seed=0)

    # Початкові параметри (theta) для калібрування
    init_theta = {"p_slip": 0.15, "step_cost": 0.10, "battery_fail_penalty": 5.0}
    twin_init = make_twin_env(twin_base_cfg, init_theta, seed=seed_actions, init_state=init_state)
    
    # Початковий replay двійника
    obs_t0, r_t0 = replay(twin_init, actions, seed=seed_actions)
    
    print("start proto:", obs_p[0,0], obs_p[0,1], "start twin:", obs_t0[0,0], obs_t0[0,1])
    m0 = compute_loss(obs_p, r_p, obs_t0, r_t0)
    print("Initial theta:", init_theta)
    print("Initial metrics:", m0)

    # Малюємо стан "ДО"
    plot_paired(
        obs_p, r_p, obs_t0, r_t0,
        title_prefix=f"BEFORE (run={run_id})",
        out_prefix=figs_dir / f"calib_slip_before_{run_id}"
    )

    # 7. Цикл Random Search
    rng = np.random.default_rng(0)
    n_trials = 400
    best = {"theta": init_theta, **m0}

    with log_path.open("w", encoding="utf-8") as f:
        # Записуємо метадані старту
        f.write(json.dumps({
            "run_id": run_id,
            "event": "start",
            "proto_cfg": asdict(proto_cfg),
            "twin_base_cfg": asdict(twin_base_cfg),
            "init_theta": init_theta,
            "init_state": init_state,
        }, ensure_ascii=False) + "\n")

        for i in range(n_trials):
            theta = sample_theta(rng)
            twin = make_twin_env(twin_base_cfg, theta, seed=seed_actions, init_state=init_state)
            obs_t, r_t = replay(twin, actions, seed=seed_actions)

            m = compute_loss(obs_p, r_p, obs_t, r_t)
            rec = {"run_id": run_id, "trial": i, **theta, **m}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if m["loss"] < best["loss"]:
                best = {"theta": theta, **m}
                print(
                    f"[best @ {i}] loss={best['loss']:.6f} "
                    f"pos_mse={best['pos_mse']:.6f} bat_mse={best['bat_mse']:.6f} "
                    f"rew_gap={best['rew_gap']:.6f} theta={theta}"
                )

        f.write(json.dumps({"run_id": run_id, "event": "end", "best": best}, ensure_ascii=False) + "\n")

    # 8. Фіналізація та малювання результату "ПІСЛЯ"
    print("\nBEST:", best)

    twin_best = make_twin_env(twin_base_cfg, best["theta"], seed=seed_actions, init_state=init_state)
    obs_tb, r_tb = replay(twin_best, actions, seed=seed_actions)

    plot_paired(
        obs_p, r_p, obs_tb, r_tb,
        title_prefix=f"AFTER (run={run_id})",
        out_prefix=figs_dir / f"calib_slip_after_{run_id}"
    )

    print(f"\nWrote log: {log_path}")
    print(f"Wrote figures to: {figs_dir}")

if __name__ == "__main__":
    main()

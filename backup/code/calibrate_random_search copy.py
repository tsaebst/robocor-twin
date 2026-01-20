from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from env import RCConfig, make_env

def sample_theta(rng):
    return {
        "p_slip": float(rng.uniform(0.0, 0.3)),
        "step_cost": float(rng.uniform(0.01, 0.5)),
        "battery_fail_penalty": float(rng.uniform(0.0, 20.0)),
    }

# -----------------------------
# Core rollout utilities
# -----------------------------

def collect_actions(env, n_steps: int, seed: int = 0) -> List[Any]:
    obs, info = env.reset(seed=seed)
    actions: List[Any] = []
    for _ in range(n_steps):
        a = env.action_space.sample()
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return actions


def replay(env, actions: List[Any], seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    obs, info = env.reset(seed=seed)
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

def compute_loss(obs_p, r_p, obs_t, r_t,
                 w_pos=1.0, w_bat=1.0, w_rew=0.1):
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))

    pos_mse = float(((obs_p[:T, 0:2] - obs_t[:T, 0:2]) ** 2).mean())
    bat_mse = float(((obs_p[:T, 7] - obs_t[:T, 7]) ** 2).mean())
    rew_gap = float(abs(r_p[:Tr].sum() - r_t[:Tr].sum()))

    loss = w_pos * pos_mse + w_bat * bat_mse + w_rew * rew_gap
    return {
        "loss": loss,
        "pos_mse": pos_mse,
        "bat_mse": bat_mse,
        "rew_gap": rew_gap,
    }


# -----------------------------
# Config helpers
# -----------------------------

def make_env_with_overrides(base_cfg: RCConfig, overrides: Dict[str, Any]):
    cfg = asdict(base_cfg)
    cfg.update(overrides)
    return make_env(RCConfig(**cfg))


# -----------------------------
# Plotting
# -----------------------------

def plot_paired(obs_p: np.ndarray, r_p: np.ndarray, obs_t: np.ndarray, r_t: np.ndarray,
                title_prefix: str, out_path: Path):
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))

    x_p, y_p = obs_p[:T, 0], obs_p[:T, 1]
    x_t, y_t = obs_t[:T, 0], obs_t[:T, 1]

    # 1) Trajectories overlay
    plt.figure()
    plt.plot(x_p, y_p, label="prototype")
    plt.plot(x_t, y_t, "--", label="twin")
    plt.title(f"{title_prefix} - Paired trajectories")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".traj.png"), dpi=160)
    plt.close()

    # 2) Battery overlay
    plt.figure()
    plt.plot(obs_p[:T, 7], label="prototype battery (obs[7])")
    plt.plot(obs_t[:T, 7], "--", label="twin battery (obs[7])")
    plt.title(f"{title_prefix} - Battery over time")
    plt.xlabel("t"); plt.ylabel("battery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".bat.png"), dpi=160)
    plt.close()

    # 3) Cumulative reward overlay
    plt.figure()
    plt.plot(np.cumsum(r_p[:Tr]), label="prototype cum reward")
    plt.plot(np.cumsum(r_t[:Tr]), "--", label="twin cum reward")
    plt.title(f"{title_prefix} - Cumulative reward")
    plt.xlabel("t"); plt.ylabel("sum reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".rew.png"), dpi=160)
    plt.close()


# -----------------------------
# Random search calibration
# -----------------------------

def sample_theta(rng: np.random.Generator) -> Dict[str, Any]:
    # Ranges chosen to be reasonable around defaults:
    # battery_max default=30; allow wider for mismatch
    battery_max = int(rng.integers(20, 301))  # [20, 300]
    # step_cost default=0.1
    step_cost = float(rng.uniform(0.01, 0.5))
    # battery_fail_penalty default=5.0
    battery_fail_penalty = float(rng.uniform(0.0, 20.0))
    return {
        "battery_max": battery_max,
        "step_cost": step_cost,
        "battery_fail_penalty": battery_fail_penalty,
    }


def main():
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True)
    figs_dir = Path("figures"); figs_dir.mkdir(exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    log_path = logs_dir / f"calib_random_search_{run_id}.jsonl"

    # -----------------------------
    # Define prototype and initial twin
    # -----------------------------
    proto_cfg = RCConfig(battery_max=200, seed=0)  # longer episode for calibration signal
    proto_env = make_env(proto_cfg)

    n_steps = 200
    seed_actions = 0

    actions = collect_actions(proto_env, n_steps=n_steps, seed=seed_actions)
    obs_p, r_p = replay(proto_env, actions, seed=seed_actions)

    # Initial twin guess: use defaults
    
    twin_base_cfg = RCConfig(seed=0)
    twin_base = make_env(RCConfig(seed=seed_actions))
	twin = SlipActionWrapper(twin_base, p_slip=theta["p_slip"], seed=seed_actions)

    init_theta = {"battery_max": twin_base_cfg.battery_max,
                  "step_cost": twin_base_cfg.step_cost,
                  "battery_fail_penalty": twin_base_cfg.battery_fail_penalty}
    twin_env_init = make_env_with_overrides(twin_base_cfg, init_theta)
    obs_t0, r_t0 = replay(twin_env_init, actions, seed=seed_actions)

    m0 = compute_loss(obs_p, r_p, obs_t0, r_t0)
    print("Initial theta:", init_theta)
    print("Initial metrics:", m0)

    plot_paired(
        obs_p, r_p, obs_t0, r_t0,
        title_prefix=f"BEFORE (run={run_id})",
        out_path=figs_dir / f"calib_before_{run_id}"
    )

    # -----------------------------
    # Random search
    # -----------------------------
    rng = np.random.default_rng(0)
    n_trials = 200  # increase to 1k+ later

    best = {"theta": init_theta, **m0}

    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "event": "start", "proto_cfg": asdict(proto_cfg)}, ensure_ascii=False) + "\n")

        for i in range(n_trials):
            theta = sample_theta(rng)
            twin_env = make_env_with_overrides(twin_base_cfg, theta)
            obs_t, r_t = replay(twin_env, actions, seed=seed_actions)

            m = compute_loss(obs_p, r_p, obs_t, r_t)
            rec = {"run_id": run_id, "trial": i, **theta, **m}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if m["loss"] < best["loss"]:
                best = {"theta": theta, **m}
                print(f"[best @ {i}] loss={best['loss']:.6f} pos_mse={best['pos_mse']:.6f} bat_mse={best['bat_mse']:.6f} rew_gap={best['rew_gap']:.6f} theta={theta}")

        f.write(json.dumps({"run_id": run_id, "event": "end", "best": best}, ensure_ascii=False) + "\n")

    print("\nBEST:", best)

    # Plot AFTER with best theta
    twin_env_best = make_env_with_overrides(twin_base_cfg, best["theta"])
    obs_tb, r_tb = replay(twin_env_best, actions, seed=seed_actions)

    plot_paired(
        obs_p, r_p, obs_tb, r_tb,
        title_prefix=f"AFTER (run={run_id})",
        out_path=figs_dir / f"calib_after_{run_id}"
    )

    print(f"\nWrote log: {log_path}")
    print(f"Wrote figures: figures/calib_before_{run_id}.* and figures/calib_after_{run_id}.*")


if __name__ == "__main__":
    main()


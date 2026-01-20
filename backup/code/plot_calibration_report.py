from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from env import RCConfig, make_env


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_actions(env, n_steps: int, seed: int = 0):
    obs, info = env.reset(seed=seed)
    actions = []
    for _ in range(n_steps):
        a = env.action_space.sample()
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return actions


def replay(env, actions, seed: int = 0):
    obs, info = env.reset(seed=seed)
    obs_list = [np.array(obs, dtype=float)]
    r_list = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(np.array(obs, dtype=float))
        r_list.append(float(r))
        if term or trunc:
            break
    return np.stack(obs_list, axis=0), np.array(r_list, dtype=float)


def compute_metrics(obs_p, r_p, obs_t, r_t, gamma_rew=0.1):
    T = min(len(obs_p), len(obs_t))
    Tr = min(len(r_p), len(r_t))

    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    rew_gap = float(abs(np.sum(r_p[:Tr]) - np.sum(r_t[:Tr])))
    loss = bat_mse + gamma_rew * rew_gap
    return {"bat_mse": bat_mse, "rew_gap": rew_gap, "loss": loss}


def best_so_far(arr):
    out = []
    best = float("inf")
    for x in arr:
        best = min(best, float(x))
        out.append(best)
    return np.array(out, dtype=float)


def main():
    run_id = "ecce9840f0"  # <-- зміни на свій, якщо інший
    log_path = Path("logs") / f"calib_random_search_{run_id}.jsonl"
    out_dir = Path("figures"); out_dir.mkdir(exist_ok=True)

    rows = load_jsonl(log_path)

    trials = [r for r in rows if "trial" in r]
    if not trials:
        raise RuntimeError("No trials found in JSONL log")

    trial_idx = np.array([t["trial"] for t in trials], dtype=int)
    loss = np.array([t["loss"] for t in trials], dtype=float)
    bat_mse = np.array([t["bat_mse"] for t in trials], dtype=float)
    rew_gap = np.array([t["rew_gap"] for t in trials], dtype=float)

    # 1) Метрики по ітераціях (raw)
    plt.figure()
    plt.plot(trial_idx, loss)
    plt.title("Calibration objective (loss) per trial")
    plt.xlabel("trial")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_dir / f"report_loss_raw_{run_id}.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(trial_idx, bat_mse)
    plt.title("Battery MSE per trial")
    plt.xlabel("trial")
    plt.ylabel("battery MSE")
    plt.tight_layout()
    plt.savefig(out_dir / f"report_batmse_raw_{run_id}.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(trial_idx, rew_gap)
    plt.title("Reward gap |Δ cumulative reward| per trial")
    plt.xlabel("trial")
    plt.ylabel("reward gap")
    plt.tight_layout()
    plt.savefig(out_dir / f"report_rewgap_raw_{run_id}.png", dpi=180)
    plt.close()

    # 2) Best-so-far (це найкраще для baseline-репорту)
    plt.figure()
    plt.plot(trial_idx, best_so_far(loss))
    plt.title("Best-so-far loss over trials")
    plt.xlabel("trial")
    plt.ylabel("best loss")
    plt.tight_layout()
    plt.savefig(out_dir / f"report_loss_bestsofar_{run_id}.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(trial_idx, best_so_far(bat_mse))
    plt.title("Best-so-far battery MSE over trials")
    plt.xlabel("trial")
    plt.ylabel("best battery MSE")
    plt.tight_layout()
    plt.savefig(out_dir / f"report_batmse_bestsofar_{run_id}.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(trial_idx, best_so_far(rew_gap))
    plt.title("Best-so-far reward gap over trials")
    plt.xlabel("trial")
    plt.ylabel("best reward gap")
    plt.tight_layout()
    plt.savefig(out_dir / f"report_rewgap_bestsofar_{run_id}.png", dpi=180)
    plt.close()

    # 3) Візуалізація траєкторій: prototype vs (init twin) vs (best twin)
    # Витягаємо init і best з логів
    start = next(r for r in rows if r.get("event") == "start")
    end = next(r for r in rows if r.get("event") == "end")
    proto_cfg = RCConfig(**start["proto_cfg"])

    # init theta беремо з дефолтів (як у calibrate_random_search.py)
    init_theta = {"battery_max": 30, "step_cost": 0.1, "battery_fail_penalty": 5.0}
    best_theta = end["best"]["theta"]

    proto = make_env(proto_cfg)
    actions = collect_actions(proto, n_steps=200, seed=0)

    obs_p, r_p = replay(proto, actions, seed=0)

    # init twin
    twin_init = make_env(RCConfig(**{**start["proto_cfg"], **init_theta, "seed": 0}))
    obs_i, r_i = replay(twin_init, actions, seed=0)

    # best twin
    twin_best = make_env(RCConfig(**{**start["proto_cfg"], **best_theta, "seed": 0}))
    obs_b, r_b = replay(twin_best, actions, seed=0)

    # overlay trajectories
    T = min(len(obs_p), len(obs_i), len(obs_b))
    plt.figure()
    plt.plot(obs_p[:T, 0], obs_p[:T, 1], label="prototype")
    plt.plot(obs_i[:T, 0], obs_i[:T, 1], "--", label="twin (init)")
    plt.plot(obs_b[:T, 0], obs_b[:T, 1], ":", label="twin (baseline best)")
    plt.title("Trajectories under identical actions (baseline calibration)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"report_trajectory_overlay_{run_id}.png", dpi=180)
    plt.close()

    # battery overlays
    plt.figure()
    plt.plot(obs_p[:T, 7], label="prototype")
    plt.plot(obs_i[:T, 7], "--", label="twin (init)")
    plt.plot(obs_b[:T, 7], ":", label="twin (baseline best)")
    plt.title("Battery trajectories under identical actions")
    plt.xlabel("t"); plt.ylabel("battery (obs[7])")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"report_battery_overlay_{run_id}.png", dpi=180)
    plt.close()

    # numeric summary (для вставки в текст)
    m_init = compute_metrics(obs_p, r_p, obs_i, r_i)
    m_best = compute_metrics(obs_p, r_p, obs_b, r_b)
    summary = {
        "run_id": run_id,
        "init_theta": init_theta,
        "best_theta": best_theta,
        "metrics_init": m_init,
        "metrics_best": m_best,
    }

    summary_path = out_dir / f"report_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote figures to:", out_dir)
    print("Wrote summary:", summary_path)
    print("Init metrics:", m_init)
    print("Best metrics:", m_best)


if __name__ == "__main__":
    main()


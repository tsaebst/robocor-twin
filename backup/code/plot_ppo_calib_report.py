# scripts/plot_calibration.py
import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def read_updates(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue

            ev = o.get("event", None)
            if ev in ("oracle_update", "oracle_update_skipped"):
                rows.append(o)
    return rows


def series(rows, keypath, default=np.nan):
    out = []
    for r in rows:
        cur = r
        ok = True
        for k in keypath:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        out.append(cur if ok else default)

    # be defensive: cast non-numeric to nan
    arr = []
    for v in out:
        try:
            arr.append(float(v))
        except Exception:
            arr.append(np.nan)
    return np.array(arr, dtype=float)


def theta_series(rows, name):
    vals = []
    for r in rows:
        te = r.get("theta_est", {}) or {}
        v = te.get(name, np.nan)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def theta_true_const(rows, name):
    # take first non-nan
    for r in rows:
        tt = r.get("theta_true", {}) or {}
        v = tt.get(name, np.nan)
        if v is None or is_nan(v):
            continue
        try:
            return float(v)
        except Exception:
            continue
    return np.nan


def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = int(w)
    if w <= 1 or len(x) == 0:
        return x
    if len(x) < w:
        return x
    k = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, k, mode="same")


def _ensure_outdir(prefix: str):
    p = Path(prefix)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_prefix", default="plots/run")
    ap.add_argument("--roll", type=int, default=25)
    args = ap.parse_args()

    _ensure_outdir(args.out_prefix)

    rows = read_updates(args.log)
    if not rows:
        raise SystemExit("No oracle_update / oracle_update_skipped events found.")

    run_id = rows[0].get("run_id", "run")

    # Prefer oracle budget as x-axis for fairness (works even with warm-up)
    t = series(rows, ["t_global"])
    u = series(rows, ["update_idx"])
    k_used = series(rows, ["k_used_total"])

    # If k_used is missing/NaN everywhere, fallback to update_idx
    x_budget = k_used
    if not np.isfinite(x_budget).any():
        x_budget = u

    # Accuracy
    pos_mse = series(rows, ["acc", "acc/pos_mse"])
    bat_mse = series(rows, ["acc", "acc/bat_mse"])
    rew_mae = series(rows, ["acc", "acc/rew_mae"])
    theta_mean_abs = series(rows, ["acc", "acc/theta_mean_abs"])

    # Loss components
    l_total = series(rows, ["loss", "loss/total"])
    l_pos = series(rows, ["loss", "loss/pos"])
    l_bat = series(rows, ["loss", "loss/bat"])
    l_rew = series(rows, ["loss", "loss/rew"])
    l_slip = series(rows, ["loss", "loss/slip"])
    l_term = series(rows, ["loss", "loss/term"])

    # Thetas
    bm = theta_series(rows, "battery_max")
    sc = theta_series(rows, "step_cost")
    ps = theta_series(rows, "p_slip")

    bm_true = theta_true_const(rows, "battery_max")
    sc_true = theta_true_const(rows, "step_cost")
    ps_true = theta_true_const(rows, "p_slip")  # likely nan

    # -----------------------------
    # Plot 1: Theta convergence
    # -----------------------------
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(x_budget, bm, label="twin_est")
    if not np.isnan(bm_true):
        ax[0].axhline(bm_true, linestyle="--", label="oracle_true")
    ax[0].set_ylabel("battery_max")
    ax[0].legend()

    ax[1].plot(x_budget, sc, label="twin_est")
    if not np.isnan(sc_true):
        ax[1].axhline(sc_true, linestyle="--", label="oracle_true")
    ax[1].set_ylabel("step_cost")
    ax[1].legend()

    ax[2].plot(x_budget, ps, label="twin_est")
    if not np.isnan(ps_true):
        ax[2].axhline(ps_true, linestyle="--", label="oracle_true")
    ax[2].set_ylabel("p_slip")
    ax[2].set_xlabel("Oracle queries used (k_used_total)")
    ax[2].legend()

    fig.suptitle(f"Theta Convergence vs Oracle Budget (run={run_id})")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_theta_budget_{run_id}.png", dpi=200)
    plt.close(fig)

    # -----------------------------
    # Plot 2: Accuracy + total loss
    # -----------------------------
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(x_budget, rolling_mean(theta_mean_abs, args.roll), label="theta_mean_abs (roll)")
    ax[0].plot(x_budget, rolling_mean(pos_mse, args.roll), label="pos_mse (roll)")
    ax[0].set_ylabel("Accuracy (lower=better)")
    ax[0].legend()

    ax[1].plot(x_budget, rolling_mean(bat_mse, args.roll), label="bat_mse (roll)")
    ax[1].plot(x_budget, rolling_mean(rew_mae, args.roll), label="rew_mae (roll)")
    ax[1].set_ylabel("Dynamics/Reward mismatch")
    ax[1].legend()

    ax[2].plot(x_budget, rolling_mean(l_total, args.roll), label="loss_total (roll)")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Oracle queries used (k_used_total)")
    ax[2].legend()

    fig.suptitle(f"Calibration Metrics vs Oracle Budget (smoothed) (run={run_id})")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_metrics_budget_{run_id}.png", dpi=200)
    plt.close(fig)

    # -----------------------------
    # Plot 3: Loss components
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_budget, rolling_mean(l_pos, args.roll), label="loss_pos")
    ax.plot(x_budget, rolling_mean(l_bat, args.roll), label="loss_bat")
    ax.plot(x_budget, rolling_mean(l_rew, args.roll), label="loss_rew")
    ax.plot(x_budget, rolling_mean(l_term, args.roll), label="loss_term")
    ax.plot(x_budget, rolling_mean(l_slip, args.roll), label="loss_slip")
    ax.set_title(f"Loss Components vs Oracle Budget (smoothed) (run={run_id})")
    ax.set_xlabel("Oracle queries used (k_used_total)")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_loss_components_budget_{run_id}.png", dpi=200)
    plt.close(fig)

    # -----------------------------
    # Plot 4: Cumulative queries over steps
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, k_used, label="cumulative queries")
    ax.set_title(f"Cumulative Queries (run={run_id})")
    ax.set_xlabel("Steps (t_global)")
    ax.set_ylabel("Queries used")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_cumulative_queries_{run_id}.png", dpi=200)
    plt.close(fig)

    # -----------------------------
    # Plot 5: Query-rate vs accuracy
    # -----------------------------
    dt = np.maximum(1.0, np.diff(t, prepend=t[0]))
    dq = np.diff(k_used, prepend=k_used[0])
    qrate = dq / dt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, rolling_mean(qrate, args.roll), label="query_rate (roll)")
    ax.set_xlabel("Steps (t_global)")
    ax.set_ylabel("Query rate")

    ax2 = ax.twinx()
    ax2.plot(t, rolling_mean(theta_mean_abs, args.roll), linestyle="--", label="theta_mean_abs (roll)")
    ax2.set_ylabel("Theta error (lower=better)")

    ax.set_title(f"Query Rate vs Theta Accuracy (run={run_id})")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_query_vs_thetaacc_{run_id}.png", dpi=200)
    plt.close(fig)

    print("Saved plots with prefix:", args.out_prefix)


if __name__ == "__main__":
    main()

# scripts/plot_query_adaptation_vs_theta_accuracy.py
# Purpose:
#   Show (1) how the RL controller adapts oracle query frequency over time
#   and (2) how twin parameter accuracy vs oracle changes on the same plot.
#
# Usage:
#   python scripts/plot_query_adaptation_vs_theta_accuracy.py \
#     --log logs/ppo_calib_<runid>.jsonl \
#     --out plots/query_adapt_vs_thetaacc_<runid>.png \
#     --roll 51
#
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _isfinite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = int(w)
    if w <= 1 or x.size == 0:
        return x
    if x.size < w:
        # still return something sensible
        k = np.ones(max(1, x.size), dtype=float) / max(1, x.size)
        return np.convolve(x, k, mode="same")
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")


def read_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            ev = o.get("event", None)
            if ev in ("oracle_update", "oracle_update_skipped", "step", "reset"):
                rows.append(o)
    return rows


def extract_updates(rows):
    # Prefer update events (they contain acc/loss/k_used_total/t_global)
    upd = [r for r in rows if r.get("event") in ("oracle_update", "oracle_update_skipped")]
    return upd


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
        if not ok or cur is None or (isinstance(cur, float) and math.isnan(cur)):
            out.append(default)
        else:
            out.append(cur)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to jsonl log file (ppo_calib_*.jsonl)")
    ap.add_argument("--out", default=None, help="Output png path. If omitted, save next to log.")
    ap.add_argument("--roll", type=int, default=51, help="Rolling window for smoothing")
    ap.add_argument("--min_dt", type=float, default=1.0, help="Min dt for rate stability")
    args = ap.parse_args()

    rows = read_rows(args.log)
    upd = extract_updates(rows)
    if not upd:
        raise SystemExit("No oracle_update / oracle_update_skipped events found in log.")

    run_id = upd[0].get("run_id", "run")

    # --- X axis: time (steps)
    t = series(upd, ["t_global"])
    # ensure monotonic-ish
    # (if any missing values, drop them)
    mask = np.isfinite(t)
    upd = [u for u, m in zip(upd, mask) if m]
    t = t[mask]

    # --- Oracle budget usage
    k_used = series(upd, ["k_used_total"])
    # drop bad
    mask = np.isfinite(k_used)
    upd = [u for u, m in zip(upd, mask) if m]
    t = t[mask]
    k_used = k_used[mask]

    # --- Accuracy of theta vs oracle (lower is better)
    theta_mean_abs = series(upd, ["acc", "acc/theta_mean_abs"])

    # If theta_mean_abs is missing for some points, keep NaNs but smooth safely
    # We’ll forward-fill for visualization consistency.
    if theta_mean_abs.size > 0:
        # forward fill NaNs
        for i in range(theta_mean_abs.size):
            if not np.isfinite(theta_mean_abs[i]):
                theta_mean_abs[i] = theta_mean_abs[i - 1] if i > 0 else np.nan
        # if still NaN at start, set to 0 for plotting
        if not np.isfinite(theta_mean_abs[0]):
            theta_mean_abs[0] = 0.0

    # --- Query rate: dq/dt between consecutive update events
    dt = np.diff(t, prepend=t[0])
    dt = np.maximum(dt, float(args.min_dt))
    dq = np.diff(k_used, prepend=k_used[0])
    qrate = dq / dt  # queries per step (at update resolution)

    qrate_s = rolling_mean(qrate, args.roll)
    theta_s = rolling_mean(theta_mean_abs, args.roll)

    # Identify whether this update actually performed oracle query
    # (both update and skipped correspond to a query in your env logic)
    did_query = np.ones_like(t, dtype=float)

    # Optional: show "where queries happen" as small markers along baseline
    # We'll place them at a small fraction of left axis.
    q_mark_y = np.nanmin(qrate_s) if np.isfinite(np.nanmin(qrate_s)) else 0.0
    q_mark_y = q_mark_y + 0.02 * (np.nanmax(qrate_s) - q_mark_y + 1e-9)

    # --- Plot
    fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.plot(t, qrate_s, label=f"query_rate (roll={args.roll})")
    ax.scatter(t[did_query > 0], np.full(np.sum(did_query > 0), q_mark_y), s=8, alpha=0.35, label="oracle query event")
    ax.set_xlabel("Steps (t_global)")
    ax.set_ylabel("Query rate (queries / step)")

    ax2 = ax.twinx()
    ax2.plot(t, theta_s, linestyle="--", label=f"theta_mean_abs (roll={args.roll})")
    ax2.set_ylabel("Theta accuracy error (lower = better)")

    ax.set_title(f"Adaptive Querying vs Twin Parameter Accuracy (run={run_id})")

    # Merge legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()

    out = args.out
    if out is None:
        p = Path(args.log)
        out = str(p.with_suffix("")) + f"_query_vs_thetaacc_{run_id}.png"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print("Saved:", out)
    print("Notes:")
    print(" - query_rate is computed as Δ(k_used_total) / Δ(t_global) between successive update events.")
    print(" - theta_mean_abs is taken from acc/theta_mean_abs; lower means closer to oracle parameters.")


if __name__ == "__main__":
    main()


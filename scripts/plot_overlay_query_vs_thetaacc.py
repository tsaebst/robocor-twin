import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = int(w)
    if w <= 1 or x.size == 0:
        return x
    if x.size < w:
        k = np.ones(max(1, x.size), dtype=float) / max(1, x.size)
        return np.convolve(x, k, mode="same")
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _series(rows, keypath, default=np.nan):
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
        if (not ok) or cur is None:
            out.append(default)
        else:
            try:
                out.append(float(cur))
            except Exception:
                out.append(default)
    return np.asarray(out, dtype=float)


def load_ppo(ppo_log: str):
    rows = _read_jsonl(ppo_log)
    upd = [r for r in rows if r.get("event") in ("oracle_update", "oracle_update_skipped")]

    if not upd:
        raise SystemExit(f"[PPO] No oracle_update events in {ppo_log}")

    t = _series(upd, ["t_global"])
    k = _series(upd, ["k_used_total"])
    th = _series(upd, ["acc", "acc/theta_mean_abs"])

    for i in range(len(th)):
        if not np.isfinite(th[i]):
            th[i] = th[i - 1] if i > 0 else 0.0

    run_id = upd[0].get("run_id", "ppo")
    return {"run_id": run_id, "t": t, "k": k, "theta_err": th}


def load_a5(a5_log: str):
    rows = _read_jsonl(a5_log)

    steps = [r for r in rows if r.get("event") == "step" and ("t_global" in r)]
    if not steps:
        raise SystemExit(f"[A5] No step events in {a5_log}")

    t = _series(steps, ["t_global"])
    k = _series(steps, ["k_used_total"])

    updates = [r for r in rows if r.get("event") == "calib_update"]
    if updates:
        ut = _series(updates, ["t_global"])
        uth = _series(updates, ["acc", "acc/theta_mean_abs"])
    else:
        ut = np.asarray([], dtype=float)
        uth = np.asarray([], dtype=float)

    theta_err = np.full_like(t, np.nan, dtype=float)
    if ut.size > 0:
        j = 0
        last = np.nan
        for i in range(len(t)):
            while j < len(ut) and ut[j] <= t[i]:
                if np.isfinite(uth[j]):
                    last = uth[j]
                j += 1
            theta_err[i] = last

    if len(theta_err) > 0 and not np.isfinite(theta_err[0]):
        theta_err[0] = 0.0
    for i in range(1, len(theta_err)):
        if not np.isfinite(theta_err[i]):
            theta_err[i] = theta_err[i - 1]

    start = next((r for r in rows if r.get("event") == "start"), {})
    run_id = start.get("run_id", "a5")
    return {"run_id": run_id, "t": t, "k": k, "theta_err": theta_err}


def compute_query_rate(t: np.ndarray, k: np.ndarray, min_dt: float = 1.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    k = np.asarray(k, dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt = np.maximum(dt, float(min_dt))
    dq = np.diff(k, prepend=k[0])
    return dq / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_log", required=True)
    ap.add_argument("--a5_log", required=True)
    ap.add_argument("--out", default="plots/overlay_query_vs_thetaacc.png")
    ap.add_argument("--roll", type=int, default=51)
    args = ap.parse_args()

    ppo = load_ppo(args.ppo_log)
    a5 = load_a5(args.a5_log)

    ppo_qr = compute_query_rate(ppo["t"], ppo["k"])
    a5_qr = compute_query_rate(a5["t"], a5["k"])

    ppo_qr_s = rolling_mean(ppo_qr, args.roll)
    a5_qr_s = rolling_mean(a5_qr, args.roll)
    ppo_th_s = rolling_mean(ppo["theta_err"], args.roll)
    a5_th_s = rolling_mean(a5["theta_err"], args.roll)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(ppo["t"], ppo_qr_s, label=f"PPO query_rate (run={ppo['run_id']})")
    ax.plot(a5["t"], a5_qr_s, label=f"A5 query_rate (run={a5['run_id']})", linestyle="--")

    ax.set_xlabel("Steps (t_global)")
    ax.set_ylabel("Query rate (queries / step)")

    ax2 = ax.twinx()
    ax2.plot(ppo["t"], ppo_th_s, label="PPO theta_mean_abs", linestyle=":")
    ax2.plot(a5["t"], a5_th_s, label="A5 theta_mean_abs", linestyle="-.")
    ax2.set_ylabel("Theta error (mean abs; lower=better)")

    ax.set_title("Overlay: Query Adaptation vs Theta Accuracy (PPO vs A5)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()

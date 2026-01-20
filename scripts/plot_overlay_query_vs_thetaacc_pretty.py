from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def apply_pub_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 260,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10,
        "lines.linewidth": 2.4,
    })


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    win = int(win)
    if win <= 1 or len(x) == 0:
        return x
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(xp, k, mode="valid")


def extract_ppo(rows: List[Dict[str, Any]]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    run_id = "ppo"
    t_list: List[int] = []
    k_list: List[int] = []
    e_list: List[float] = []

    for r in rows:
        ev = r.get("event")
        if ev not in ("oracle_update", "oracle_update_skipped"):
            continue

        if r.get("run_id"):
            run_id = str(r.get("run_id"))

        tg = r.get("t_global")
        ku = r.get("k_used_total")
        if tg is None or ku is None:
            continue

        acc = r.get("acc", {}) if isinstance(r.get("acc", {}), dict) else {}
        theta_ma = acc.get("acc/theta_mean_abs", np.nan)

        t_list.append(int(tg))
        k_list.append(int(ku))
        try:
            e_list.append(float(theta_ma))
        except Exception:
            e_list.append(np.nan)

    if len(t_list) < 3:
        return run_id, np.array([]), np.array([]), np.array([])

    t = np.asarray(t_list, dtype=float)
    k = np.asarray(k_list, dtype=float)
    e = np.asarray(e_list, dtype=float)

    # rate between update points
    dt = np.diff(t, prepend=t[0])
    dk = np.diff(k, prepend=k[0])
    dt[dt <= 0] = np.nan
    qrate = dk / dt
    qrate[~np.isfinite(qrate)] = 0.0

    # ffill theta error
    if len(e) > 0 and not np.isfinite(e[0]):
        e[0] = 0.0
    for i in range(1, len(e)):
        if not np.isfinite(e[i]):
            e[i] = e[i - 1]

    return run_id, t, qrate, e


def extract_a5(rows: List[Dict[str, Any]]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    run_id = "a5"
    # A5 has step-level logging with cumulative queries
    steps = [r for r in rows if r.get("event") == "step"]
    if not steps:
        return run_id, np.array([]), np.array([]), np.array([])

    if rows and isinstance(rows[0], dict) and rows[0].get("run_id"):
        run_id = str(rows[0]["run_id"])

    t = np.asarray([int(r.get("t_global", i)) for i, r in enumerate(steps)], dtype=float)
    k = np.asarray([int(r.get("k_used_total", 0)) for r in steps], dtype=float)

    # query rate per step 
    dt = np.diff(t, prepend=t[0])
    dq = np.diff(k, prepend=k[0])
    dt[dt <= 0] = 1.0
    qrate = dq / dt
    qrate[~np.isfinite(qrate)] = 0.0

    # theta error lives in calib_update
    updates = [r for r in rows if r.get("event") == "calib_update"]
    e = np.full_like(t, np.nan, dtype=float)

    if updates:
        ut = np.asarray([int(r.get("t_global", 0)) for r in updates], dtype=float)
        ue = []
        for r in updates:
            acc = r.get("acc", {}) if isinstance(r.get("acc", {}), dict) else {}
            ue.append(acc.get("acc/theta_mean_abs", np.nan))
        ue = np.asarray([float(x) if x is not None else np.nan for x in ue], dtype=float)

        j = 0
        last = np.nan
        for i in range(len(t)):
            while j < len(ut) and ut[j] <= t[i]:
                if np.isfinite(ue[j]):
                    last = ue[j]
                j += 1
            e[i] = last

    # ffill
    if len(e) > 0 and not np.isfinite(e[0]):
        e[0] = 0.0
    for i in range(1, len(e)):
        if not np.isfinite(e[i]):
            e[i] = e[i - 1]

    return run_id, t, qrate, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_log", required=True)
    ap.add_argument("--a5_log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roll", type=int, default=51)
    args = ap.parse_args()

    apply_pub_style()

    ppo_rows = read_jsonl(args.ppo_log)
    a5_rows = read_jsonl(args.a5_log)

    ppo_id, ppo_t, ppo_qr, ppo_e = extract_ppo(ppo_rows)
    a5_id, a5_t, a5_qr, a5_e = extract_a5(a5_rows)

    if len(ppo_t) == 0:
        raise SystemExit(f"[PPO] No oracle_update/oracle_update_skipped events in {args.ppo_log}")
    if len(a5_t) == 0:
        raise SystemExit(f"[A5] No step events in {args.a5_log}")

    # smooth
    ppo_qr_s = rolling_mean(ppo_qr, args.roll)
    ppo_e_s = rolling_mean(ppo_e, args.roll)
    a5_qr_s = rolling_mean(a5_qr, args.roll)
    a5_e_s = rolling_mean(a5_e, args.roll)

    # Colors: colorblind-friendly-ish
    c_ppo = "#1f77b4"   # blue
    c_a5 = "#ff7f0e"    # orange

    fig, ax = plt.subplots(figsize=(12.8, 5.2))

    # Left axis query rate
    ax.plot(ppo_t, ppo_qr_s, color=c_ppo, label=f"PPO query rate (run={ppo_id})")
    ax.plot(a5_t, a5_qr_s, color=c_a5, linestyle="--", label=f"A5 query rate (run={a5_id})")

    ax.set_xlabel("Steps (t_global)")
    ax.set_ylabel("Query rate (queries / step)")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.3f}"))

    # Right axis theta error
    ax2 = ax.twinx()
    ax2.plot(ppo_t, ppo_e_s, color=c_ppo, linestyle=":", label="PPO theta error (mean abs)")
    ax2.plot(a5_t, a5_e_s, color=c_a5, linestyle=":", label="A5 theta error (mean abs)")
    ax2.set_ylabel("Theta error (mean abs, lower is better)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

    title = "Adaptive Oracle Querying vs Calibration Accuracy (PPO vs A5)"
    subtitle = f"PPO log: {Path(args.ppo_log).name} | A5 log: {Path(args.a5_log).name}"
    ax.set_title(title + "\n" + subtitle)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", str(out.resolve()))


if __name__ == "__main__":
    main()

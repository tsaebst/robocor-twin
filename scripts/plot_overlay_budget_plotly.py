from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def ffill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    if not np.isfinite(x[0]):
        x[0] = 0.0
    for i in range(1, len(x)):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    return x


def extract_ppo_budget_series(rows: List[Dict[str, Any]]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (run_id, k_used_total, query_rate, theta_mean_abs)
    computed at oracle_update / oracle_update_skipped points.
    """
    run_id = "ppo"
    k_list: List[int] = []
    t_list: List[int] = []
    e_list: List[float] = []

    for r in rows:
        ev = r.get("event")
        if ev not in ("oracle_update", "oracle_update_skipped"):
            continue

        if r.get("run_id"):
            run_id = str(r.get("run_id"))

        ku = r.get("k_used_total")
        tg = r.get("t_global")
        if ku is None or tg is None:
            continue

        acc = r.get("acc", {}) if isinstance(r.get("acc", {}), dict) else {}
        theta_ma = acc.get("acc/theta_mean_abs", np.nan)

        k_list.append(int(ku))
        t_list.append(int(tg))
        try:
            e_list.append(float(theta_ma))
        except Exception:
            e_list.append(np.nan)

    if len(k_list) < 3:
        return run_id, np.array([]), np.array([]), np.array([])

    k = np.asarray(k_list, dtype=float)
    t = np.asarray(t_list, dtype=float)
    e = ffill_nan(np.asarray(e_list, dtype=float))

    # query_rate between update points: delta_k / delta_t
    dk = np.diff(k, prepend=k[0])
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.nan
    qr = dk / dt
    qr[~np.isfinite(qr)] = 0.0

    return run_id, k, qr, e


def extract_a5_budget_series(rows: List[Dict[str, Any]]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    A5 logs step-level: event='step' has t_global and k_used_total.
    We build a series over k_used_total by grouping steps that share the same k.
    Returns (run_id, k_used_total_unique, mean_query_rate, theta_mean_abs_ffill).
    """
    run_id = "a5"
    if rows and rows[0].get("run_id"):
        run_id = str(rows[0]["run_id"])

    steps = [r for r in rows if r.get("event") == "step"]
    if not steps:
        return run_id, np.array([]), np.array([]), np.array([])

    t = np.asarray([int(r.get("t_global", i)) for i, r in enumerate(steps)], dtype=float)
    k = np.asarray([int(r.get("k_used_total", 0)) for r in steps], dtype=float)

    # per-step query indicator: dq 
    dq = np.diff(k, prepend=k[0])
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1.0
    qr_step = dq / dt
    qr_step[~np.isfinite(qr_step)] = 0.0
    updates = [r for r in rows if r.get("event") == "calib_update"]
    e_step = np.full_like(t, np.nan, dtype=float)

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
            e_step[i] = last

    e_step = ffill_nan(e_step)

    # group by k 
    # for each k value, average query_rate over the segment, and take last theta error
    k_int = k.astype(int)
    uniq_k = np.unique(k_int)

    qr_by_k = np.zeros_like(uniq_k, dtype=float)
    e_by_k = np.zeros_like(uniq_k, dtype=float)

    for idx, kk in enumerate(uniq_k):
        mask = (k_int == kk)
        qr_by_k[idx] = float(np.mean(qr_step[mask])) if np.any(mask) else 0.0
        e_by_k[idx] = float(e_step[np.where(mask)[0][-1]]) if np.any(mask) else 0.0

    return run_id, uniq_k.astype(float), qr_by_k, e_by_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_log", required=True)
    ap.add_argument("--a5_log", required=True)
    ap.add_argument("--out", required=True, help="Output path .html or .png (png requires kaleido)")
    ap.add_argument("--roll", type=int, default=21)
    ap.add_argument("--max_k", type=int, default=None, help="Optional: clip to this oracle budget")
    args = ap.parse_args()

    ppo_rows = read_jsonl(args.ppo_log)
    a5_rows = read_jsonl(args.a5_log)

    ppo_id, ppo_k, ppo_qr, ppo_e = extract_ppo_budget_series(ppo_rows)
    a5_id, a5_k, a5_qr, a5_e = extract_a5_budget_series(a5_rows)

    if len(ppo_k) == 0:
        raise SystemExit(f"[PPO] No oracle_update/oracle_update_skipped with k_used_total & t_global in {args.ppo_log}")
    if len(a5_k) == 0:
        raise SystemExit(f"[A5] No step events with k_used_total in {args.a5_log}")

    # Smooth
    ppo_qr_s = rolling_mean(ppo_qr, args.roll)
    ppo_e_s  = rolling_mean(ppo_e,  args.roll)
    a5_qr_s  = rolling_mean(a5_qr,  max(3, args.roll // 2))
    a5_e_s   = rolling_mean(a5_e,   max(3, args.roll // 2))

    # Clip to common max_k (so curves share comparable budget range)
    max_k_common = int(min(np.max(ppo_k), np.max(a5_k)))
    if args.max_k is not None:
        max_k_common = int(min(max_k_common, int(args.max_k)))

    def clip_series(k, y1, y2):
        m = k <= max_k_common
        return k[m], y1[m], y2[m]

    ppo_kc, ppo_qrc, ppo_ec = clip_series(ppo_k, ppo_qr_s, ppo_e_s)
    a5_kc,  a5_qrc,  a5_ec  = clip_series(a5_k,  a5_qr_s,  a5_e_s)

    # Professional plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Colors 
    c_ppo = "#2F6BFF"  # vivid blue
    c_a5  = "#FF7A00"  # warm orange

    fig.add_trace(
        go.Scatter(
            x=ppo_kc, y=ppo_qrc,
            mode="lines",
            name=f"PPO query rate (run={ppo_id})",
            line=dict(color=c_ppo, width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=a5_kc, y=a5_qrc,
            mode="lines",
            name=f"A5 query rate (run={a5_id})",
            line=dict(color=c_a5, width=3, dash="dash"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ppo_kc, y=ppo_ec,
            mode="lines",
            name="PPO theta error (mean abs)",
            line=dict(color=c_ppo, width=2, dash="dot"),
            opacity=0.95,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=a5_kc, y=a5_ec,
            mode="lines",
            name="A5 theta error (mean abs)",
            line=dict(color=c_a5, width=2, dash="dot"),
            opacity=0.95,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Adaptive Oracle Querying vs Calibration Accuracy (PPO vs A5)<br>"
                 f"<sup>Aligned by oracle budget (k_used_total), clipped to K={max_k_common}</sup>",
            x=0.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=70, r=70, t=90, b=60),
        width=1400,
        height=520,
    )

    fig.update_xaxes(
        title_text="Oracle budget used (k_used_total)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Query rate (queries / step)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Theta error (mean abs, lower is better)",
        showgrid=False,
        zeroline=False,
        secondary_y=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".html":
        fig.write_html(str(out), include_plotlyjs="cdn")
        print("Saved:", str(out.resolve()))
    elif out.suffix.lower() == ".png":
        # requires kaleido
        fig.write_image(str(out), scale=2)
        print("Saved:", str(out.resolve()))
    else:
        # default: html
        out2 = out.with_suffix(".html")
        fig.write_html(str(out2), include_plotlyjs="cdn")
        print("Saved:", str(out2.resolve()))


if __name__ == "__main__":
    main()

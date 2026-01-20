import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _get(d: dict, keypath: List[str], default=np.nan):
    cur = d
    for k in keypath:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


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


def compute_query_rate(t: np.ndarray, k: np.ndarray, min_dt: float = 1.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    k = np.asarray(k, dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt = np.maximum(dt, float(min_dt))
    dk = np.diff(k, prepend=k[0])
    return dk / dt


def forward_fill(x: np.ndarray, fill0: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if not np.isfinite(x[0]):
        x[0] = float(fill0)
    for i in range(1, len(x)):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    return x


# Loaders

def load_ppo(path: str) -> Dict[str, np.ndarray]:
    rows = _read_jsonl(path)
    upd = [r for r in rows if r.get("event") in ("oracle_update", "oracle_update_skipped")]
    if not upd:
        raise SystemExit(f"[PPO] No oracle_update events in {path}")

    t = np.asarray([float(r.get("t_global", np.nan)) for r in upd], dtype=float)
    k = np.asarray([float(r.get("k_used_total", np.nan)) for r in upd], dtype=float)
    th = np.asarray([float(_get(r, ["acc", "acc/theta_mean_abs"], np.nan)) for r in upd], dtype=float)

    th = forward_fill(th, fill0=0.0)
    run_id = upd[0].get("run_id", "ppo")
    return {"name": f"PPO (run={run_id})", "t": t, "k": k, "theta_err": th}


def load_baseline_step_log(path: str, label: str) -> Dict[str, np.ndarray]:
    rows = _read_jsonl(path)
    steps = [r for r in rows if r.get("event") == "step" and ("t_global" in r)]
    if not steps:
        raise SystemExit(f"[{label}] No step events with t_global in {path}")

    t = np.asarray([float(r.get("t_global", np.nan)) for r in steps], dtype=float)
    k = np.asarray([float(r.get("k_used_total", np.nan)) for r in steps], dtype=float)
    th = np.asarray([float(_get(r, ["acc", "acc/theta_mean_abs"], np.nan)) for r in steps], dtype=float)
    th = forward_fill(th, fill0=0.0)

    run_id = None
    start = next((r for r in rows if r.get("event") == "start"), None)
    if start is not None:
        run_id = start.get("run_id", None)
    if run_id is None:
        run_id = Path(path).stem

    return {"name": f"{label} (run={run_id})", "t": t, "k": k, "theta_err": th}


def load_a5_flexible(path: str, label: str = "A5") -> Dict[str, np.ndarray]:
    rows = _read_jsonl(path)

    # Try normal step-style first
    steps = [r for r in rows if r.get("event") == "step" and ("t_global" in r)]
    if steps:
        t = np.asarray([float(r.get("t_global", np.nan)) for r in steps], dtype=float)
        k = np.asarray([float(r.get("k_used_total", np.nan)) for r in steps], dtype=float)
        th = np.asarray([float(_get(r, ["acc", "acc/theta_mean_abs"], np.nan)) for r in steps], dtype=float)
        th = forward_fill(th, fill0=0.0)
    else:
        # episode_summary fallback
        eps = [r for r in rows if r.get("event") == "episode_summary"]
        if not eps:
            raise SystemExit(f"[{label}] No step events and no episode_summary in {path}")

        # sort by episode index if present
        eps.sort(key=lambda r: int(r.get("ep", 0)))

        # Build cumulative timeline
        t_list = [0.0]
        k_list = [0.0]
        th_list = [np.nan]

        t_cur = 0.0
        k_cur = 0.0

        for r in eps:
            ep_len = int(r.get("episode_length", 0))
            q = int(r.get("oracle_queries", 0))

            # optional theta acc
            th_val = _get(r, ["acc", "acc/theta_mean_abs"], np.nan)
            try:
                th_val = float(th_val) if th_val is not None else np.nan
            except Exception:
                th_val = np.nan

            # advance
            t_cur += max(0, ep_len)
            k_cur += max(0, q)

            t_list.append(float(t_cur))
            k_list.append(float(k_cur))
            th_list.append(th_val)

        t = np.asarray(t_list, dtype=float)
        k = np.asarray(k_list, dtype=float)
        th = forward_fill(np.asarray(th_list, dtype=float), fill0=0.0)

    start = next((r for r in rows if r.get("event") == "start"), {})
    run_id = start.get("run_id", Path(path).stem)
    return {"name": f"{label} (run={run_id})", "t": t, "k": k, "theta_err": th}


# CLI parsing
def parse_baselines(items: List[str]) -> List[Tuple[str, str]]:
    out = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Bad --baseline '{it}'. Expected format Name=path/to/log.jsonl")
        name, path = it.split("=", 1)
        name = name.strip()
        path = path.strip()
        out.append((name, path))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_log", required=True)
    ap.add_argument("--baseline", action="append", default=[], help='Format: "A1=logs/a1_run_x.jsonl" (repeatable)')
    ap.add_argument("--out", default="plots/overlay_multi.png")
    ap.add_argument("--roll", type=int, default=51)
    args = ap.parse_args()

    series = []

    # PPO
    ppo = load_ppo(args.ppo_log)
    series.append(ppo)

    # Baselines
    baselines = parse_baselines(args.baseline)
    for label, path in baselines:
        if label.strip().upper() == "A5":
            s = load_a5_flexible(path, label="A5")
        else:
            s = load_baseline_step_log(path, label=label)
        series.append(s)

    # Plot
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))

    # left axis query rate
    for s in series:
        qr = compute_query_rate(s["t"], s["k"])
        qr_s = rolling_mean(qr, args.roll)
        style = "-" if s["name"].startswith("PPO") else "--"
        ax.plot(s["t"], qr_s, linestyle=style, label=f"{s['name']} query rate")

    ax.set_xlabel("Steps (t_global)")
    ax.set_ylabel("Query rate (Δk / Δt)")

    # right axis theta error
    ax2 = ax.twinx()
    for s in series:
        th_s = rolling_mean(s["theta_err"], args.roll)
        style = ":" if s["name"].startswith("PPO") else "-."
        ax2.plot(s["t"], th_s, linestyle=style, label=f"{s['name']} theta error")

    ax2.set_ylabel("Theta error (mean abs; lower is better)")

    ax.set_title("Oracle query adaptation vs calibration accuracy (PPO + A1..A5)")

    # unified legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()

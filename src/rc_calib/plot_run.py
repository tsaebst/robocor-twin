from __future__ import annotations

import argparse
from pathlib import Path

from rc_calib.plotting import (
    load_jsonl_run,
    plot_trajectory_overlay,
    plot_battery_with_queries_and_updates,
    plot_drift_dynamics,
    plot_theta_traces,
    plot_one_panel_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to JSONL log produced by runner")
    ap.add_argument("--outdir", type=str, default="figures", help="Output directory for figures")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run = load_jsonl_run(log_path)

    p1 = outdir / f"{run.policy}_{run.run_id}.traj_overlay.png"
    p2 = outdir / f"{run.policy}_{run.run_id}.battery_queries_updates.png"
    p3 = outdir / f"{run.policy}_{run.run_id}.drift_on_queries.png"
    p4 = outdir / f"{run.policy}_{run.run_id}.theta_traces.png"
    p5 = outdir / f"{run.policy}_{run.run_id}.summary_1panel.png"

    # separate plots (saved, not auto-open)
    plot_trajectory_overlay(run, p1)
    plot_battery_with_queries_and_updates(run, p2)
    plot_drift_dynamics(run, p3)
    plot_theta_traces(run, p4)

    # combined summary (saved + auto-open on macOS)
    plot_one_panel_summary(run, p5)

    print("WROTE:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)
    print(" -", p4)
    print(" -", p5)


if __name__ == "__main__":
    main()

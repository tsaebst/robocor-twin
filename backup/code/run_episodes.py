from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from env import RoboCourierEnv  # later: RCConfig, make_env


def to_jsonable(x: Any) -> Any:
    """Safely convert numpy/scalars to JSON-serializable types."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if is_dataclass(x):
        return asdict(x)
    return x


def run_episode(env: RoboCourierEnv, max_steps: int = 500, seed: int | None = None) -> Dict[str, Any]:
    if seed is not None:
        reset_out = env.reset(seed=seed)
    else:
        reset_out = env.reset()

    obs, info = reset_out
    total_reward = 0.0
    steps = 0

    # Log a few basic diagnostics about observation/action spaces
    meta = {
        "obs_shape": getattr(obs, "shape", None),
        "obs_dtype": str(getattr(obs, "dtype", "")),
        "action_space": str(getattr(env, "action_space", "")),
        "timestamp_start": time.time(),
    }

    traj = []  # per-step logs

    for t in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        traj.append({
            "t": t,
            "action": to_jsonable(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs": to_jsonable(next_obs),
            "info": to_jsonable(info),
        })

        if terminated or truncated:
            obs = next_obs
            break

        obs = next_obs

    summary = {
        "steps": steps,
        "total_reward": total_reward,
        "done": bool(traj[-1]["terminated"] or traj[-1]["truncated"]) if traj else False,
        "timestamp_end": time.time(),
    }

    return {"meta": meta, "summary": summary, "trajectory": traj}


def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    out_path = out_dir / f"robocourier_random_{run_id}.jsonl"

    env = RoboCourierEnv()

    n_episodes = 5
    max_steps = 300

    print(f"Writing logs to: {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            ep_data = run_episode(env, max_steps=max_steps, seed=ep)
            record = {
                "run_id": run_id,
                "episode": ep,
                **ep_data["meta"],
                **ep_data["summary"],
            }
            # Store only summary in JSONL (compact); trajectory to separate file if needed
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Also write full trajectory as JSON (optional, but useful for plots)
            traj_path = out_dir / f"robocourier_random_{run_id}_ep{ep}.json"
            traj_path.write_text(
                json.dumps(ep_data, ensure_ascii=False),
                encoding="utf-8",
            )

            print(f"ep={ep} steps={record['steps']} total_reward={record['total_reward']:.3f} done={record['done']}")

    print("Done.")


if __name__ == "__main__":
    main()


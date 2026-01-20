from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

from env import RCConfig, make_env
from rc_calib.wrappers import SlipActionWrapper
from rc_calib.interfaces import Policy, Calibrator, Theta, TwinSample, OracleSample


# ---------- oracle http client ----------
import requests

class RoboCourierOracle:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def reset(self, seed: int = 0, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"seed": int(seed), "config_overrides": config_overrides}
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def step(self, session_id: str, action: int) -> Dict[str, Any]:
        payload = {"session_id": session_id, "action": int(action)}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


# ---------- per-episode sync helpers ----------
def unwrap_env(env_obj: Any, max_depth: int = 20) -> Any:
    base = env_obj
    for _ in range(max_depth):
        if hasattr(base, "env"):
            base = getattr(base, "env")
        else:
            break
    return base

def force_world_state_on_env(env_obj: Any, state: Dict[str, Any]) -> None:
    base = unwrap_env(env_obj)
    base.rx, base.ry = int(state["rx"]), int(state["ry"])
    base.px, base.py = int(state["px"]), int(state["py"])
    base.dx, base.dy = int(state["dx"]), int(state["dy"])
    base.cx, base.cy = int(state["cx"]), int(state["cy"])
    base.battery = int(state["battery"])
    base.has_package = bool(state["has_package"])

class EpisodeInitWrapper:
    def __init__(self, env_obj: Any, init_state: Dict[str, Any]):
        self.env = env_obj
        self.init_state = init_state
        self.action_space = env_obj.action_space

    def reset(self, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        force_world_state_on_env(self.env, self.init_state)
        base = unwrap_env(self.env)
        if hasattr(base, "_obs") and callable(getattr(base, "_obs")):
            obs = base._obs()
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: Any):
        return self.env.step(action)


def make_twin_env(theta: Theta, init_state: Dict[str, Any], seed: int) -> Any:
    cfg = RCConfig(
        grid_size=int(theta["grid_size"]),
        battery_max=int(theta["battery_max"]),
        step_cost=float(theta["step_cost"]),
        delivery_reward=float(theta["delivery_reward"]),
        battery_fail_penalty=float(theta["battery_fail_penalty"]),
        use_stay=bool(theta["use_stay"]),
        seed=int(seed),
    )
    base = make_env(cfg)
    epi = EpisodeInitWrapper(base, init_state=init_state)
    twin = SlipActionWrapper(epi, p_slip=float(theta["p_slip"]), seed=int(seed))
    return twin


# ---------- JSONL logger ----------
class JSONL:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# ---------- Runner ----------
def run_baseline(
    *,
    oracle_url: str,
    policy: Policy,
    calibrator: Calibrator,
    theta0: Theta,
    seed0: int,
    n_episodes: int,
    h_total: int,
    oracle_budget: int,
    oracle_sleep_s: float,
    oracle_query_period: int,              # e.g. every 5 steps, subject to budget
    update_every_n_queries: int,            # e.g. update theta each 10 oracle queries
    apply_updates_mid_episode: bool,        # False = safest, True = show dynamics in same episode
    out_dir: Path,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    log_path = out_dir / "logs" / f"{policy.name}_{run_id}.jsonl"
    logger = JSONL(log_path)

    oracle = RoboCourierOracle(oracle_url)

    theta = dict(theta0)
    calibrator.reset(theta)

    used_oracle_steps = 0
    used_queries = 0
    t_global = 0

    logger.write({
        "event": "run_start",
        "run_id": run_id,
        "policy": policy.name,
        "calibrator": calibrator.name,
        "oracle_url": oracle_url,
        "seed0": seed0,
        "n_episodes": n_episodes,
        "h_total": h_total,
        "oracle_budget": oracle_budget,
        "oracle_query_period": oracle_query_period,
        "update_every_n_queries": update_every_n_queries,
        "apply_updates_mid_episode": apply_updates_mid_episode,
        "theta0": theta0,
    })

    ep_summaries: List[Dict[str, Any]] = []

    for episode_id in range(n_episodes):
        if t_global >= h_total:
            break

        seed = seed0 + episode_id

        # reset oracle for this episode
        oreset = oracle.reset(seed=seed)
        sid = oreset["session_id"]
        obs_oracle0 = np.asarray(oreset["obs"], dtype=np.float32)
        ws0 = oreset["world_state"]

        # build twin env synced to oracle init_state
        twin = make_twin_env(theta, init_state=ws0, seed=seed)
        obs_twin, _ = twin.reset(seed=seed)

        policy.reset(episode_id=episode_id, seed=seed)

        logger.write({
            "event": "episode_start",
            "run_id": run_id,
            "episode_id": episode_id,
            "seed": seed,
            "theta": dict(theta),
            "oracle_init_state": ws0,
            "oracle_obs0": obs_oracle0.tolist(),
            "t_global": t_global,
        })

        ep_t = 0
        ep_oracle_queries = 0

        # local buffers for episode metrics
        drift_pos: List[float] = []
        drift_bat: List[float] = []

        while t_global < h_total:
            # choose action from twin obs
            a = int(policy.act(np.asarray(obs_twin, dtype=np.float32), theta))

            # always step twin
            obs_twin2, r_twin, term_t, trunc_t, _ = twin.step(a)
            obs_twin2 = np.asarray(obs_twin2, dtype=np.float32)

            logger.write({
                "event": "twin_step",
                "run_id": run_id,
                "episode_id": episode_id,
                "t_global": t_global,
                "ep_t": ep_t,
                "action": a,
                "obs_twin": obs_twin2.tolist(),
                "reward_twin": float(r_twin),
            })

            # decide if we query oracle now
            do_query = False
            if used_oracle_steps < oracle_budget:
                if (t_global % oracle_query_period) == 0:
                    do_query = True
                # allow policy to request extra queries (still budgeted)
                if policy.wants_oracle(obs_twin2, theta, t_global=t_global, ep_t=ep_t):
                    do_query = True

            if do_query and used_oracle_steps < oracle_budget:
                ostep = oracle.step(sid, a)
                used_oracle_steps += 1
                used_queries += 1
                ep_oracle_queries += 1

                obs_oracle = np.asarray(ostep["obs"], dtype=np.float32)
                r_oracle = float(ostep["reward"])
                ws = ostep["world_state"]

                logger.write({
                    "event": "oracle_query",
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "t_global": t_global,
                    "ep_t": ep_t,
                    "action": a,
                    "obs_oracle": obs_oracle.tolist(),
                    "reward_oracle": r_oracle,
                    "world_state_oracle": ws,
                    "used_oracle_steps": used_oracle_steps,
                })

                # record drift on query points (for plots)
                drift_pos.append(float(np.mean((obs_oracle[0:2] - obs_twin2[0:2]) ** 2)))
                drift_bat.append(float((obs_oracle[7] - obs_twin2[7]) ** 2))

                # send sample to calibrator
                calibrator.observe(
                    TwinSample(t_global=t_global, episode_id=episode_id, ep_t=ep_t, action=a,
                             obs_twin=obs_twin2, reward_twin=float(r_twin)),
                    OracleSample(t_global=t_global, episode_id=episode_id, ep_t=ep_t, action=a,
                                obs_oracle=obs_oracle, reward_oracle=r_oracle, world_state=ws),
                )

                if oracle_sleep_s > 0:
                    time.sleep(float(oracle_sleep_s))

                # update theta if due
                if (used_queries % update_every_n_queries) == 0 and calibrator.ready_to_update():
                    new_theta, summary = calibrator.update(theta)
                    theta = dict(new_theta)

                    logger.write({
                        "event": "theta_update",
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "t_global": t_global,
                        "ep_t": ep_t,
                        "new_theta": dict(theta),
                        "summary_metrics": dict(summary),
                        "used_queries": used_queries,
                    })

                    if apply_updates_mid_episode:
                        # rebuild twin env *mid-episode* (risky but shows dynamics)
                        # keep current oracle world_state as the new init state
                        twin = make_twin_env(theta, init_state=ws, seed=seed)
                        obs_twin2, _ = twin.reset(seed=seed)

            # advance
            obs_twin = obs_twin2
            ep_t += 1
            t_global += 1

            # stop if twin episode ends
            if term_t or trunc_t:
                break

        ep_summary = {
            "episode_id": episode_id,
            "seed": seed,
            "t_global_end": t_global,
            "ep_len": ep_t,
            "ep_oracle_queries": ep_oracle_queries,
            "drift_pos_mse_mean_on_queries": float(np.mean(drift_pos)) if drift_pos else None,
            "drift_bat_mse_mean_on_queries": float(np.mean(drift_bat)) if drift_bat else None,
            "theta_end": dict(theta),
        }
        ep_summaries.append(ep_summary)

        logger.write({
            "event": "episode_end",
            "run_id": run_id,
            **ep_summary,
        })

    logger.write({
        "event": "run_end",
        "run_id": run_id,
        "used_oracle_steps": used_oracle_steps,
        "used_queries": used_queries,
        "episodes": ep_summaries,
        "theta_final": dict(theta),
    })
    logger.close()

    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "theta_final": theta,
        "episodes": ep_summaries,
    }


import numpy as np
import matplotlib.pyplot as plt

from env import RCConfig, make_env
from rc_calib.twin import RoboCourierTwin
import sys
from pathlib import Path

# Add <project_root>/src to import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import RCConfig, make_env
from rc_calib.wrappers import SlipActionWrapper

# prototype (no slip)
proto = make_env(RCConfig(seed=0))

# twin with slip
twin_base = make_env(RCConfig(seed=0))
twin = SlipActionWrapper(twin_base, p_slip=0.15, seed=0)


def collect_actions(env, n_steps=100, seed=0):
    obs, info = env.reset(seed=seed)
    actions = []
    for _ in range(n_steps):
        a = env.action_space.sample()
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return actions


def replay(env, actions, seed=0):
    obs, info = env.reset(seed=seed)
    obs_list = [obs]
    r_list = []
    done_flags = []

    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(obs)
        r_list.append(r)
        done_flags.append(term or trunc)
        if term or trunc:
            break

    obs_arr = np.array(obs_list, dtype=float)
    r_arr = np.array(r_list, dtype=float) if r_list else np.array([], dtype=float)
    return obs_arr, r_arr


def main():
    # Prototype
    proto = make_env(RCConfig(battery_max=200, seed=0))
    actions = collect_actions(proto, n_steps=200, seed=0)

    # Twin with different theta (для демонстрації різниці)
    twin = RoboCourierTwin(theta={"battery_max": 120, "step_cost": 0.2}, seed=0)

    obs_p, r_p = replay(proto, actions, seed=0)
    obs_t, r_t = replay(twin.env, actions, seed=0)

    # XY map
    plt.figure()
    plt.plot(obs_p[:, 0], obs_p[:, 1], label="prototype")
    plt.plot(obs_t[:, 0], obs_t[:, 1], "--", label="twin")
    plt.title("Paired trajectories under identical actions")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.show()

    # Battery over time (obs[7])
    T = min(len(obs_p), len(obs_t))
    plt.figure()
    plt.plot(obs_p[:T, 7], label="prototype battery")
    plt.plot(obs_t[:T, 7], "--", label="twin battery")
    plt.title("Battery over time (obs[7])")
    plt.xlabel("t"); plt.ylabel("battery")
    plt.legend()
    plt.show()

    # Reward comparison
    Tr = min(len(r_p), len(r_t))
    plt.figure()
    plt.plot(np.cumsum(r_p[:Tr]), label="prototype cumulative reward")
    plt.plot(np.cumsum(r_t[:Tr]), "--", label="twin cumulative reward")
    plt.title("Cumulative reward under identical actions")
    plt.xlabel("t"); plt.ylabel("sum reward")
    plt.legend()
    plt.show()

    # Simple discrepancy metrics
    pos_mse = float(np.mean((obs_p[:T, :2] - obs_t[:T, :2]) ** 2))
    bat_mse = float(np.mean((obs_p[:T, 7] - obs_t[:T, 7]) ** 2))
    print("pos_mse:", pos_mse)
    print("bat_mse:", bat_mse)

if __name__ == "__main__":
    main()


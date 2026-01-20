import numpy as np
import matplotlib.pyplot as plt

from env import RCConfig, make_env

def run(env, n_steps=200, seed=0):
    obs, info = env.reset(seed=seed)
    traj = []

    for t in range(n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        traj.append((t, obs, r, term, trunc, info))
        if term or trunc:
            break

    return traj

def main():
    # Довгий епізод
    cfg = RCConfig(battery_max=200, seed=0)
    env = make_env(cfg)

    traj = run(env, n_steps=300, seed=0)

    obs = np.array([x[1] for x in traj], dtype=float)
    r = np.array([x[2] for x in traj], dtype=float)
    term = np.array([x[3] for x in traj], dtype=bool)
    trunc = np.array([x[4] for x in traj], dtype=bool)

    x = obs[:, 0]
    y = obs[:, 1]
    battery = obs[:, 7]  # з твого графіка: obs[7] монотонно спадає

    # 1) “Карта”: траєкторія + колір батареї
    plt.figure()
    sc = plt.scatter(x, y, c=battery, s=50)
    plt.plot(x, y, linewidth=1)
    plt.scatter(x[0], y[0], s=120, label="start")
    plt.scatter(x[-1], y[-1], s=120, label="end")
    plt.colorbar(sc, label="battery (obs[7])")
    plt.title(f"Long episode trajectory (steps={len(traj)}, terminated={term.any()}, truncated={trunc.any()})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.show()

    # 2) Окремо: battery(t)
    t = np.arange(len(traj))
    plt.figure()
    plt.plot(t, battery)
    plt.title("Battery over time")
    plt.xlabel("t"); plt.ylabel("battery (obs[7])")
    plt.show()

    # 3) Окремо: reward(t) і cumulative reward
    plt.figure()
    plt.plot(t, r)
    plt.title("Reward over time")
    plt.xlabel("t"); plt.ylabel("reward")
    plt.show()

    plt.figure()
    plt.plot(t, np.cumsum(r))
    plt.title("Cumulative reward")
    plt.xlabel("t"); plt.ylabel("sum reward")
    plt.show()

if __name__ == "__main__":
    main()

import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# знайдемо останній ep0 файл
p = sorted(glob.glob("logs/robocourier_random_*_ep0.json"))[-1]
d = json.load(open(p, "r", encoding="utf-8"))

obs = np.array([step["obs"] for step in d["trajectory"]], dtype=float)

plt.figure()
plt.plot(obs[:, 0], obs[:, 1], marker="o", label="trajectory")
plt.scatter(obs[0, 0], obs[0, 1], c="green", s=80, label="start")
plt.scatter(obs[-1, 0], obs[-1, 1], c="red", s=80, label="end")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Agent trajectory")
plt.show()


import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CalibEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_seeds,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_seeds = list(eval_seeds)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if (self.num_timesteps % self.eval_freq) != 0:
            return True

        # Run evaluation episodes on held-out seeds
        for i in range(self.n_eval_episodes):
            seed = int(self.eval_seeds[(i + self.num_timesteps // self.eval_freq) % len(self.eval_seeds)])
            obs, _ = self.eval_env.reset(seed=seed)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, trunc, info = self.eval_env.step(action)
                done = bool(done or trunc)

            # Optional: log per-episode summary (last state already logged by env on each oracle_update)

        return True


from __future__ import annotations
import numpy as np

class SlipActionWrapper:
    """
    With probability p_slip, replace intended action with a random action.
    This induces controlled stochastic transition dynamics in the twin.
    """
    def __init__(self, env, p_slip: float, seed: int = 0):
        self.env = env
        self.p_slip = float(p_slip)
        self.rng = np.random.default_rng(seed)

        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, **kwargs):
    	env = gym.make("Pendulum-v1")
    	u = env.unwrapped
    	u.g = 12.5
    	u.max_torque = 1.6
    	# b у Gym Pendulum зазвичай немає; якщо хочеш b — тоді треба робити wrapper або власний env-клон.
    	return self.env.reset(**kwargs)

    def step(self, action):
        a = action
        if self.rng.random() < self.p_slip:
            a = self.action_space.sample()
        return self.env.step(a)

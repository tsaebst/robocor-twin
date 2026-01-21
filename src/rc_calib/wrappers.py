from __future__ import annotations
import numpy as np
import gymnasium as gym  # Додано імпорт!

class SlipActionWrapper:
    """
    With probability p_slip, replace intended action with a random action.
    This induces controlled stochastic transition dynamics in the twin.
    """
    def __init__(self, env, p_slip: float, seed: int = 0):
        self.env = env
        self.p_slip = float(p_slip)
        self.rng = np.random.default_rng(seed)

        # Копіюємо простори з оригінального середовища
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, **kwargs):
        # ВАЖЛИВО: Навіщо тут Pendulum? 
        # Якщо ви хочете змінити параметри Pendulum, це має сенс лише якщо self.env є Pendulum.
        # Якщо ви просто хочете скинути основне середовище:
        return self.env.reset(**kwargs)

    def step(self, action):
        a = action
        # Логіка проковзування (slip)
        if self.rng.random() < self.p_slip:
            a = self.action_space.sample()
        
        # Виконуємо крок у вкладеному середовищі
        return self.env.step(a)
    
    # Додамо метод render, щоб уникнути помилок, якщо PPO захоче візуалізацію
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
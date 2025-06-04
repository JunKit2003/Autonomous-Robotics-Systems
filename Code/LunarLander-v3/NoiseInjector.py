# NoiseInjector.py
import gymnasium as gym
import numpy as np

class NoiseInjector(gym.Wrapper):
    def __init__(self, env, gravity=None, wind_power=None, turbulence_power=None):
        super(NoiseInjector, self).__init__(env)
        self.gravity = gravity
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.fluctuations = []  # List to store fluctuations during an episode

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_static_noise()  # Apply static noise (e.g., gravity) at reset
        self.fluctuations = []  # Reset fluctuations at the start of each episode
        return obs, info

    def step(self, action):
        # Apply dynamic noise before the step
        self._apply_dynamic_noise()

        # Step through the environment with the given action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return obs, reward, terminated, truncated, info

    def _apply_static_noise(self):
        # Access the base environment
        env = self.env.unwrapped

        # Adjust gravity if specified (applied once at reset)
        if self.gravity is not None:
            if hasattr(env, 'world') and hasattr(env.world, 'gravity'):
                env.world.gravity = (0, self.gravity)

        # Set initial wind power if specified
        if self.wind_power is not None and hasattr(env, 'wind_power'):
            env.wind_power = self.wind_power

    def _apply_dynamic_noise(self):
        # Access the base environment
        env = self.env.unwrapped

        # Introduce dynamic wind fluctuations using turbulence power
        if self.turbulence_power is not None:
            if hasattr(env, 'wind_power'):
                fluctuation = np.random.uniform(-self.turbulence_power, self.turbulence_power)
                env.wind_power += fluctuation
                # Record the fluctuation
                self.fluctuations.append((fluctuation, env.wind_power))

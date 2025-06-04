# NoiseInjector.py
# NoiseInjector modifies environment parameters to inject noise into the dynamics.
# For CartPole, this might mean changing gravity or other physical parameters.
class NoiseInjector:
    def __init__(self, env_noise_params={}):
        self.env_noise_params = env_noise_params  # Dictionary of environment parameters to modify

    def inject_env_noise(self, env):
        # Inject noise by setting environment attributes to new values if provided
        for param, value in self.env_noise_params.items():
            if hasattr(env, param):
                setattr(env, param, value)
            elif hasattr(env, 'env') and hasattr(env.env, param):
                setattr(env.env, param, value)
        return env

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from utilities.visualiser import Visualiser

class SymmetryRestorationEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), max_steps=50, render_mode=None):
        super(SymmetryRestorationEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # State space: The grid itself, a 10x10 grayscale matrix
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(grid_size[0] * grid_size[1],), dtype=np.float32)  # Flattened space

        # Action space: (x, y, up/down)
        self.action_space = spaces.MultiDiscrete([grid_size[0], grid_size[1], 2])  # x, y, direction

        # Initialize grid and CNN autoencoder loss function
        self.grid = np.random.uniform(0, 1, size=grid_size).astype(np.float32)
        self.autoencoder = self._load_pretrained_autoencoder()

    def _load_pretrained_autoencoder(self):
        """
        Mocked CNN autoencoder loss function. Replace this with the actual pretrained model.
        """
        return lambda grid: np.mean((grid - grid[::-1, ::-1])**2)  # Mock symmetry loss

    def reset(self, seed=None, **kwargs):
        """Reset the environment to start a new episode."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # Set the seed for reproducibility
        self.grid = np.random.uniform(0, 1, size=self.grid_size).astype(np.float32)
        self.current_step = 0
        return self.grid.flatten(), {}  # Flatten the grid to 1D vector

    def step(self, action):
        """
        Perform the action: modify the grid based on the (x, y, up/down) action.
        """
        x, y, direction = action
        if direction == 0: # uhhhh doesn't this bias towards non-zero numbers?
            self.grid[x, y] = max(0, self.grid[x, y] - 0.1)  # Subtract 0.1
        else:
            self.grid[x, y] = min(1, self.grid[x, y] + 0.1)  # Add 0.1

        self.current_step += 1

        # Calculate reward and determine if episode is done
        loss = self.autoencoder(self.grid)
        reward = -loss  # Lower loss means better symmetry
        done = False
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            done = True  # End the episode after max_steps
            terminated = True  # Explicitly mark it as terminated when max steps reached

        return self.grid.flatten(), reward.__float__(), terminated, truncated, {}  # Return 5 values

    def render(self, mode=None):
        """Visualize the grid."""
        if mode is None:
            mode = self.render_mode  # Use the default render_mode if not provided
        if mode == "human":
            Visualiser.visualise(self.grid)
        elif mode == "ansi":
            return str(self.grid)  # Return the grid as a string
        elif mode == "rgb_array":
            raise NotImplementedError("rgb_array mode is not implemented.")
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

# Define and check the environment
env = SymmetryRestorationEnv(render_mode="human")
check_env(env)

# Wrap the environment for PPO
env = DummyVecEnv([lambda: env])

# Define and train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Test the trained agent
obs = env.reset()
max_steps = env.get_attr("max_steps")[0]
for step in range(max_steps):  # Use max_steps directly here
    action, _states = model.predict(obs)
    obs, reward, done, truncated = env.step(action)
    env.render()
    if done:
        print(f"Episode finished with reward: {reward}")
        break
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch

from utilities.visualiser import Visualiser

class SymmetryEnv(gym.Env):
    def __init__(self):
        super(SymmetryEnv, self).__init__()

        # Hardcoded 2x2 grid with values between 0 and 1
        self.grid = np.random.uniform(0, 1, size=(2, 2))

        # Action space: 8 actions (4 cells, each with increase (+1) or decrease (-1))
        self.action_space = spaces.Discrete(8)

        # Observation space: 2x2 grid with values between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32)

    def reset(self):
        # Reset the grid to a random configuration
        self.grid = np.random.uniform(0, 1, size=(2, 2))
        return self.grid

    def step(self, action):
        # Determine which cell to modify and by how much
        cell_idx = action // 2  # Choose the cell (0-3)
        direction = 1 if action % 2 == 0 else -1  # 0 = increase, 1 = decrease

        # Modify the selected cell by 0.1
        self.grid[cell_idx // 2, cell_idx % 2] += direction * 0.1
        self.grid = np.clip(self.grid, 0, 1)  # Ensure values stay within [0, 1]

        # Calculate the reward (symmetry metric)
        reward = self.calculate_symmetry(self.grid)

        done = False  # Task is ongoing until we define a condition

        if reward > 0.98:
            done = True
            reward = 1
        else:
            reward = 0

        return self.grid, reward, done, {}

    def calculate_symmetry(self, grid):
        # Perfect symmetry metric
        # Calculate the difference between diagonally opposite cells (you can adjust this metric)
        return np.mean(np.abs(grid[0, 0] - grid[1, 1]) + np.abs(grid[0, 1] - grid[1, 0]))

    # Rendering function (leave this empty to fill in later)
    def render(self, mode='human'):
        # Fill in with any rendering logic you like (e.g., print the grid, visualize symmetry)
        Visualiser.visualise(grid=self.grid)


# Create the environment
env = DummyVecEnv([lambda: SymmetryEnv()])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Test the trained model
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    Visualiser.visualise(grid=torch.Tensor(obs).view(2, 2))
    print(f"Grid: {obs}, Reward: {reward}")
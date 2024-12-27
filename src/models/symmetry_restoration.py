import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time
from sklearn.metrics import r2_score
import hashlib
import json
import os

class SymmetryExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim * 2)

        # Get grid dimensions from observation space
        self.height, self.width = observation_space.shape

        self.policy_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.height * self.width, features_dim)
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.height * self.width, features_dim)
        )

    def forward(self, observations):
        x = observations.view(-1, 1, self.height, self.width).float()
        return torch.cat([self.policy_layers(x), self.value_conv(x)], dim=1)


class SymmetryEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'height': 3,
            'width': 4,
            'perfect_reward': 100.0,
            'step_penalty': -1.0,
            'max_steps': 10,
            'partial_reward_weight': 5.0,
            'redundant_move_penalty': -2.0
        }
        if config:
            self.config.update(config)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.config['height'], self.config['width']),
            dtype=np.int8
        )

        self.action_space = spaces.MultiDiscrete([
            self.config['height'] * self.config['width'],
            2
        ])

    @staticmethod
    def get_config_hash(config):
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @staticmethod
    def get_model_path(config):
        return f"saved_models/model_{SymmetryEnv.get_config_hash(config)}.zip"

    def calculate_symmetry_score(self):
        if not isinstance(self.grid, np.ndarray):
            raise ValueError("Grid must be a numpy array")
        if self.grid.ndim != 2:
            raise ValueError("Grid must be 2D")

        reflected_grid = self.grid[:, ::-1]
        original_flat = self.grid.flatten()
        reflected_flat = reflected_grid.flatten()

        symmetry_score = r2_score(original_flat, reflected_flat)

        return (symmetry_score + 3) / 4

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.grid = np.random.randint(
            0, 2,
            (self.config['height'], self.config['width']),
            dtype=np.int8
        )
        self.steps = 0
        self.initial_symmetry = self.calculate_symmetry_score()
        return self.grid, {}

    def render(self):
        pass

    def is_symmetric(self):
        return self.calculate_symmetry_score() == 1

    def get_reward(self):
        if self.is_symmetric():
            return self.config['perfect_reward']

        current_symmetry = self.calculate_symmetry_score()
        symmetry_improvement = current_symmetry - self.initial_symmetry
        return symmetry_improvement * self.config['partial_reward_weight'] + self.config['step_penalty']

    def step(self, action):
        pos, value = action
        row = pos // self.config['width']
        col = pos % self.config['width']

        redundant = self.grid[row, col] == value

        self.grid[row, col] = value
        reward = self.get_reward()

        if redundant:
            reward += self.config['redundant_move_penalty']

        self.steps += 1
        terminated = self.is_symmetric() or self.steps >= self.config['max_steps']

        if terminated and not self.is_symmetric():
            reward -= self.config['perfect_reward'] * 0.5

        return self.grid, reward, terminated, False, {}

def train_agent(env_config=None, load_if_exists=True):
    print("Provided environment configuration:")
    print(env_config)

    env = DummyVecEnv([lambda: SymmetryEnv(env_config)])
    model_path = SymmetryEnv.get_model_path(env_config)

    if load_if_exists and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        return PPO.load(model_path, env=env)

    policy_kwargs = {
        'features_extractor_class': SymmetryExtractor,
        'features_extractor_kwargs': {'features_dim': 64},
        'net_arch': dict(pi=[64], vf=[64])
    }

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1
    )

    model.learn(total_timesteps=env_config['learning_total_timesteps'])
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model

def demonstrate_agent(model, env_config=None, episodes=5):
    env = SymmetryEnv(env_config)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}")
        obs, _ = env.reset()
        print(f"Initial grid:\n{obs}")
        print(f"Initial symmetry: {env.calculate_symmetry_score():.2f}")

        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            done = terminated

            print(f"\nAction: pos={action[0]}, val={action[1]}")
            print(f"Grid:\n{obs}")
            print(f"Symmetry: {env.calculate_symmetry_score():.2f}")
            print(f"Reward: {reward:.2f}")
            time.sleep(0.5)

        print(f"Total reward: {total_reward:.2f}")
        print(f"Symmetric: {env.is_symmetric()}")


if __name__ == "__main__":
    env_config = {
        'height': 10,
        'width': 10,
        'perfect_reward': 10000.0,
        'step_penalty': -1.0,
        'partial_reward_weight': 5.0,
        'max_steps': 100,
        'redundant_move_penalty': -2.0,
        'learning_total_timesteps': 5000000
    }

    model = train_agent(env_config, load_if_exists=True)
    demonstrate_agent(model, env_config)
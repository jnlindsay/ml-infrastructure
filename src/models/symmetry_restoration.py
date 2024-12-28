import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
import time
from sklearn.metrics import r2_score
import hashlib
import json
import os
import sys
from typing import Any, Dict, Tuple, Union
import mlflow
from utilities.visualiser import Visualiser

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

        self.prev_symmetry = None
        self.visited_pos = set()

    @staticmethod
    def get_config_hash(config):
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @staticmethod
    def get_model_path(config):
        return f"saved_models/model_{SymmetryEnv.get_config_hash(config)}.zip"

    @property
    def curr_symmetry(self) -> float:
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
        self.prev_symmetry = None
        self.visited_pos = set()
        return self.grid, {}

    def render(self):
        pass

    def is_symmetric(self):
        return self.curr_symmetry == 1

    def get_reward(self):
        if self.is_symmetric():
            return self.config['perfect_reward']

        if self.prev_symmetry is not None:
            symmetry_improvement = self.curr_symmetry - self.prev_symmetry
        else:
            symmetry_improvement = 0

        return symmetry_improvement * self.config['partial_reward_weight'] + self.config['step_penalty']

    def step(self, action):
        pos, value = action
        row = pos // self.config['width']
        col = pos % self.config['width']

        revisited = pos in self.visited_pos
        self.visited_pos.add(pos)

        redundant = self.grid[row, col] == value

        self.grid[row, col] = value

        # calculate symmetry and reward
        reward = self.get_reward()
        self.prev_symmetry = self.curr_symmetry

        if redundant:
            reward += self.config['redundant_move_penalty']
        if revisited:
            reward += self.config['revisited_penalty']

        self.steps += 1

        terminated = False
        if self.is_symmetric() or self.steps >= self.config['max_steps']:
            terminated = True
        if len(self.visited_pos) < self.steps - self.config['allowed_revisits']:
            terminated = True
            reward -= self.config['perfect_reward'] * 0.5

        if terminated and not self.is_symmetric():
            reward -= self.config['perfect_reward'] * 0.5

        return self.grid, reward, terminated, False, {}

def train_agent(env_config=None, load_if_exists=True, loggers=None):
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

    mlflow.set_tracking_uri(uri="http://localhost:8080")
    with mlflow.start_run():
        model = PPO(
            'MlpPolicy',
            env,
            ent_coef=env_config['training_ent_coef'],
            learning_rate=env_config['learning_rate'],
            verbose=2
        )

        mlflow.log_params(env_config)

        if loggers:
            model.set_logger(loggers)

        model.learn(total_timesteps=env_config['learning_total_timesteps'])
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return model

def demonstrate_agent(model, env_config=None, episodes=5):
    env = SymmetryEnv(env_config)

    for episode in range(episodes):
        print(f"\n----------------------------------")
        print(f"\nEpisode {episode + 1}")
        obs, _ = env.reset()
        print(f"Initial grid:")
        Visualiser.visualise(obs, (2, 2))
        print(f"Initial symmetry: {env.curr_symmetry:.2f}")

        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            done = terminated

            print(f"\nAction: pos={action[0]}, val={action[1]}")
            print(f"Grid:")
            Visualiser.visualise(obs, (2, 2))
            print(f"Symmetry: {env.curr_symmetry:.2f}")
            print(f"Reward: {reward:.2f}")

        print(f"Total reward: {total_reward:.2f}")
        print(f"Symmetric: {env.is_symmetric()}")

class MLflowOutputFormat(KVWriter):
    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if key in [
                'time/iterations',
                'time/time_elapsed',
                'time/total_timesteps',
                'train/clip_range',
                'train/n_updates'
            ]:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
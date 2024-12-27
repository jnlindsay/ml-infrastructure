import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time


class SymmetryExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim * 2)

        self.policy_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9, features_dim)
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9, features_dim)
        )

    def forward(self, observations):
        x = observations.view(-1, 1, 3, 3).float()
        return torch.cat([self.policy_layers(x), self.value_conv(x)], dim=1)


class SymmetryEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'grid_size': 3,
            'perfect_reward': 100.0,
            'step_penalty': -1.0,
            'max_steps': 10,
            'partial_reward_weight': 5.0
        }
        if config:
            self.config.update(config)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.config['grid_size'], self.config['grid_size']),
            dtype=np.int8
        )

        self.action_space = spaces.MultiDiscrete([
            self.config['grid_size'] * self.config['grid_size'],
            2
        ])

    def calculate_symmetry_score(self):
        left_col = self.grid[:, 0]
        right_col = self.grid[:, -1]
        return np.sum(left_col == right_col) / len(left_col)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.grid = np.random.randint(0, 2, (self.config['grid_size'], self.config['grid_size']), dtype=np.int8)
        self.steps = 0
        self.initial_symmetry = self.calculate_symmetry_score()
        return self.grid, {}

    def render(self):
        pass

    def is_symmetric(self):
        return np.array_equal(self.grid[:, 0], self.grid[:, -1])

    def get_reward(self):
        if self.is_symmetric():
            return self.config['perfect_reward']

        current_symmetry = self.calculate_symmetry_score()
        symmetry_improvement = current_symmetry - self.initial_symmetry
        return symmetry_improvement * self.config['partial_reward_weight'] + self.config['step_penalty']

    def step(self, action):
        pos, value = action
        row, col = pos // self.config['grid_size'], pos % self.config['grid_size']

        self.grid[row, col] = value
        reward = self.get_reward()

        self.steps += 1
        terminated = self.is_symmetric() or self.steps >= self.config['max_steps']

        if terminated and not self.is_symmetric():
            reward -= self.config['perfect_reward'] * 0.5

        return self.grid, reward, terminated, False, {}


def train_agent(env_config=None):
    env = DummyVecEnv([lambda: SymmetryEnv(env_config)])

    policy_kwargs = {
        'features_extractor_class': SymmetryExtractor,
        'features_extractor_kwargs': {'features_dim': 64},
        'net_arch': dict(pi=[64], vf=[64])
    }

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=1.0,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    model.learn(total_timesteps=100000)
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

            env.render()
            time.sleep(0.5)

        print(f"Total reward: {total_reward:.2f}")
        print(f"Symmetric: {env.is_symmetric()}")


if __name__ == "__main__":
    env_config = {
        'perfect_reward': 100.0,
        'step_penalty': -1.0,
        'partial_reward_weight': 5.0,
        'max_steps': 15
    }

    model = train_agent(env_config)
    demonstrate_agent(model, env_config)
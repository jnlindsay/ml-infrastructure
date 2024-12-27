import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import json
import time


class SymmetryEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'grid_size': 3,
            'perfect_reward': 100.0,
            'step_penalty': -1.0,
            'redundant_penalty': -2.0,
            'max_steps': 100
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

        self.reset()

    def reset(self):
        # Generate random initial grid
        self.grid = np.random.randint(0, 2, (self.config['grid_size'], self.config['grid_size']), dtype=np.int8)
        self.steps = 0
        return self.grid

    def render(self, mode='human'):
        print(self.grid)

    def is_symmetric(self):
        return np.array_equal(self.grid[:, 0], self.grid[:, -1])

    def get_reward(self, pos, value):
        row, col = pos // self.config['grid_size'], pos % self.config['grid_size']

        if self.grid[row, col] == value:
            return self.config['redundant_penalty']

        if self.is_symmetric():
            return self.config['perfect_reward']

        return self.config['step_penalty']

    def step(self, action):
        pos, value = action
        row, col = pos // self.config['grid_size'], pos % self.config['grid_size']

        old_value = self.grid[row, col]
        self.grid[row, col] = value

        reward = self.get_reward(pos, value)

        self.steps += 1
        done = self.is_symmetric() or self.steps >= self.config['max_steps']

        return self.grid, reward, done, {}


def create_env(config=None):
    return SymmetryEnv(config)


def get_model_path(env_config, training_config):
    # Simple hash of configs
    config_str = json.dumps({**env_config, **training_config}, sort_keys=True)
    return f"symmetry_model_{hash(config_str)}.zip"


def train_or_load_agent(env_config=None, training_config=None):
    model_path = get_model_path(env_config or {}, training_config or {})

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        return PPO.load(model_path)

    print("Training new model...")
    default_training_config = {
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
    }
    if training_config:
        default_training_config.update(training_config)

    env = DummyVecEnv([lambda: create_env(env_config)])

    model = PPO('MlpPolicy', env,
                learning_rate=default_training_config['learning_rate'],
                n_steps=default_training_config['n_steps'],
                batch_size=default_training_config['batch_size'],
                n_epochs=default_training_config['n_epochs'],
                gamma=default_training_config['gamma'],
                verbose=1)

    model.learn(total_timesteps=default_training_config['total_timesteps'])
    model.save(model_path)
    return model


def demonstrate_agent(model, env_config=None, episodes=2):
    env = create_env(env_config)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}")
        obs = env.reset()
        print("Initial grid:")
        print(obs)
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            print(f"\nAction: position={action[0]}, value={action[1]}")
            print("Grid after action:")
            print(obs)
            print(f"Reward: {reward}")

            env.render()  # Will be implemented by user
            time.sleep(0.5)  # Add delay for visibility

        print(f"Episode finished with total reward: {total_reward}")
        print(f"Final grid is symmetric: {env.is_symmetric()}")


if __name__ == "__main__":
    env_config = {
        'perfect_reward': 20.0,
        'max_steps': 15
    }

    training_config = {
        'total_timesteps': 50000,
        'learning_rate': 1e-4
    }

    model = train_or_load_agent(env_config, training_config)
    demonstrate_agent(model, env_config)
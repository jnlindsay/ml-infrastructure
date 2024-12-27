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
        # Default configuration with adjusted rewards
        self.config = {
            'grid_size': 3,
            'perfect_reward': 100.0,  # Significantly increased perfect symmetry reward
            'step_penalty': -1.0,
            'max_steps': 10,
            'partial_reward_weight': 5.0  # Weight for partial symmetry rewards
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

    def calculate_symmetry_score(self):
        """
        Calculate what fraction of cells are symmetric about the middle column.
        Returns a value between 0 and 1.
        """
        left_col = self.grid[:, 0]
        right_col = self.grid[:, -1]
        matching_cells = np.sum(left_col == right_col)
        return matching_cells / len(left_col)

    def reset(self):
        # Generate random initial grid
        self.grid = np.random.randint(0, 2, (self.config['grid_size'],
                                             self.config['grid_size']), dtype=np.int8)
        self.steps = 0
        # Store initial symmetry score to measure improvement
        self.initial_symmetry = self.calculate_symmetry_score()
        return self.grid

    def render(self, mode='human'):
        pass  # To be implemented by user

    def is_symmetric(self):
        return np.array_equal(self.grid[:, 0], self.grid[:, -1])

    def get_reward(self):
        # Perfect symmetry reward
        if self.is_symmetric():
            return self.config['perfect_reward']

        # Partial symmetry reward
        current_symmetry = self.calculate_symmetry_score()
        symmetry_improvement = current_symmetry - self.initial_symmetry
        partial_reward = symmetry_improvement * self.config['partial_reward_weight']

        # Step penalty is always applied
        return partial_reward + self.config['step_penalty']

    def step(self, action):
        pos, value = action
        row, col = pos // self.config['grid_size'], pos % self.config['grid_size']

        # Apply action
        self.grid[row, col] = value

        # Calculate reward
        reward = self.get_reward()

        # Update step counter and check if done
        self.steps += 1
        done = self.is_symmetric() or self.steps >= self.config['max_steps']

        # Add episode termination penalty if we run out of steps without symmetry
        if done and not self.is_symmetric():
            reward -= self.config['perfect_reward'] * 0.5  # Penalty for failing to achieve symmetry

        return self.grid, reward, done, {}


def create_env(config=None):
    return SymmetryEnv(config)


def get_model_path(env_config, training_config):
    config_str = json.dumps({**env_config, **training_config}, sort_keys=True)
    return f"symmetry_model_{hash(config_str)}.zip"


def train_or_load_agent(env_config=None, training_config=None, force_train=False):
    model_path = get_model_path(env_config or {}, training_config or {})

    if os.path.exists(model_path) and not force_train:
        print(f"Loading existing model from {model_path}")
        return PPO.load(model_path)

    print("Training new model...")
    # Modified training configuration for better exploration
    default_training_config = {
        'total_timesteps': 100000,
        'learning_rate': 5e-5,  # Lower learning rate for more stable learning
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'ent_coef': 0.01,  # Increased entropy coefficient for better exploration
        'clip_range': 0.2,
        'max_grad_norm': 0.5,  # Added gradient clipping for stability
        'vf_coef': 0.5,  # Balanced value function loss
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
                ent_coef=default_training_config['ent_coef'],
                clip_range=default_training_config['clip_range'],
                max_grad_norm=default_training_config['max_grad_norm'],
                vf_coef=default_training_config['vf_coef'],
                verbose=1)

    model.learn(total_timesteps=default_training_config['total_timesteps'])
    model.save(model_path)
    return model


def demonstrate_agent(model, env_config=None, episodes=5):
    env = create_env(env_config)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}")
        obs = env.reset()
        print("Initial grid:")
        print(obs)
        print(f"Initial symmetry score: {env.calculate_symmetry_score():.2f}")

        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            print(f"\nAction: position={action[0]}, value={action[1]}")
            print("Grid after action:")
            print(obs)
            print(f"Current symmetry score: {env.calculate_symmetry_score():.2f}")
            print(f"Step reward: {reward:.2f}")

            env.render()
            time.sleep(0.5)

        print(f"Episode finished with total reward: {total_reward:.2f}")
        print(f"Final grid is symmetric: {env.is_symmetric()}")


if __name__ == "__main__":
    env_config = {
        'perfect_reward': 100.0,  # Increased perfect symmetry reward
        'step_penalty': -1.0,
        'partial_reward_weight': 5.0,
        'max_steps': 15
    }

    training_config = {
        'total_timesteps': 100000,
        'learning_rate': 5e-5,
        'ent_coef': 0.01
    }

    # Force retrain with new parameters
    model = train_or_load_agent(env_config, training_config, force_train=True)
    demonstrate_agent(model, env_config)
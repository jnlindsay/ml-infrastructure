import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from utilities.visualiser import Visualiser
from models.grid_autoencoder import GridEncoderStableBaselines
from trainers.grid_autoencoder_trainer import GridAutoencoderTrainer
from sklearn.metrics import r2_score
import torch

class SymmetryRestorationEnv(gym.Env):
    def __init__(self, grid_size=(3, 3), max_steps=81, render_mode=None):
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
        self.autoencoder_trainer = self._load_pretrained_autoencoder_trainer()

    def _load_pretrained_autoencoder_trainer(self):
        training_phases = [
            GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 100),
            GridAutoencoderTrainer.TrainingPhase("random_symmetrical", 1000, 100)
        ]

        grid_autoencoder_trainer = GridAutoencoderTrainer(self.grid_size, self.grid_size)
        grid_autoencoder_trainer.train(training_phases, force_retrain=False)

        return grid_autoencoder_trainer


    def _load_pretrained_autoencoder(self):
        """
        CNN autoencoder loss function. Replace this with the actual pretrained model.
        """

        # TODO: do we need a better way to specify models?
        training_phases = [
            GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 100),
            GridAutoencoderTrainer.TrainingPhase("random_symmetrical", 1000, 100)
        ]

        # TODO: rename to .train_or_load() or something
        grid_autoencoder_trainer = GridAutoencoderTrainer(self.grid_size, self.grid_size)
        grid_autoencoder_trainer.train(training_phases, force_retrain=False)

        def loss_calculator(grid):
            grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
            reconstructed_grid = grid_autoencoder_trainer.model.forward(grid_tensor)

            return r2_score(
                reconstructed_grid.squeeze().squeeze().detach().numpy(),
                grid_tensor.squeeze().squeeze().detach().numpy()
            )

        def calculate_symmetry(grid):
            """
            Calculate the symmetry of a 2D grid about the vertical axis using R² score.

            Args:
                grid (np.ndarray): A 2D numpy array with values in the range [0, 1].

            Returns:
                float: Symmetry score (R²) in the range [-inf, 1]. Higher means more symmetrical.
            """
            if not isinstance(grid, np.ndarray):
                raise ValueError("Input must be a numpy array.")
            if grid.ndim != 2:
                raise ValueError("Input grid must be 2D.")
            if not np.all((0 <= grid) & (grid <= 1)):
                raise ValueError("All grid values must lie in the range [0, 1].")

            # Reflect the grid about the vertical axis
            reflected_grid = grid[:, ::-1]

            # Flatten both grids to 1D for comparison
            original_flat = grid.flatten()
            reflected_flat = reflected_grid.flatten()

            thingo = grid.tolist()

            # Calculate R² score
            symmetry_score = r2_score(original_flat, reflected_flat)
            return (symmetry_score + 3) / 4

        return calculate_symmetry

    def reset(self, seed=None, **kwargs):
        """Reset the environment to start a new episode."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # Set the seed for reproducibility
        self.grid = np.random.choice([0, 1], size=self.grid_size).astype(np.float32)
        self.current_step = 0

        return self.grid.flatten(), {}  # Flatten the grid to 1D vector

    def step(self, action):
        """
        Perform the action: modify the grid based on the (x, y, up/down) action.
        """
        x, y, direction = action

        cell_original_state = self.grid[x, y].copy()

        if direction == 0: # uhhhh doesn't this bias towards non-zero numbers?
            self.grid[x, y] = max(0, self.grid[x, y] - 1)
        else:
            self.grid[x, y] = min(1, self.grid[x, y] + 1)

        self.current_step += 1

        # Calculate reward and determine if episode is done
        symmetricalness = self.autoencoder(self.grid)
        done = False
        terminated = False
        truncated = False

        print("Symmetricalness is:", symmetricalness)

        reward_payout = 10

        if symmetricalness == 1:
            # reward = reward_payout if symmetricalness > 0.90 else -1
            # reward  = (symmetricalness * 2 - 1) ** 3
            reward = reward_payout
        # elif cell_original_state == self.grid[x, y]:
        #     reward = -10
        else:
            reward = -1

        if self.current_step >= self.max_steps or reward >= reward_payout:
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
# env = VecNormalize(env)

training_phases = [
    GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 100),
    GridAutoencoderTrainer.TrainingPhase("random_symmetrical", 1000, 100)
]

grid_autoencoder_trainer = GridAutoencoderTrainer(10, 10)
grid_autoencoder_trainer.train(training_phases, force_retrain=False)

# use grid autoencoder
policy_kwargs = {
    "features_extractor_class": GridEncoderStableBaselines,
    "features_extractor_kwargs": {
        "pretrained_model": grid_autoencoder_trainer.model,
        "features_dim": 128
    }
}

# Define and train the PPO agent
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model = PPO(
    "MlpPolicy",
    env,
    batch_size=64,
    learning_rate=0.003,
    verbose=1
)
model.learn(total_timesteps=100000)

# Test the trained agent
obs = env.reset()
max_steps = env.get_attr("max_steps")[0]
for step in range(max_steps):  # Use max_steps directly here
    action, _states = model.predict(obs)
    obs, reward, done, truncated = env.step(action)
    print("Reward has been turned into:", reward)
    env.render()
    if done:
        print(f"Episode finished with reward: {reward}")
        break
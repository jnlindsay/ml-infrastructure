from utilities.grid import GridFactory
from trainers.trainer import Trainer
from models.grid_counter import GridCounter
import torch
import torch.nn as nn
import random

class GridCounterTrainer(Trainer):
    def __init__(self, input_size, hidden_size, grid_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        super().__init__("grid_counter")

    def model_factory(self):
        return GridCounter(
            self.input_size,
            self.hidden_size,
            self.grid_size
        )

    def generate_training_set(self, batch_size):
        grid_factory = GridFactory(self.grid_size, self.grid_size)

        grids = []
        counts = []
        for num_dots in range(batch_size):
            num_dots = random.randint(0, 10)
            grid = grid_factory.generate_random_spaced_dots(num_dots=num_dots)
            grids.append(grid.unsqueeze(0))
            counts.append(num_dots)

        grids_batch = torch.stack(grids)
        counts_batch = torch.tensor(counts, dtype=torch.float32)

        return grids_batch, counts_batch

    def train(self, force_retrain=False):
        if self.save_file_exists() and not force_retrain:
            print(f"This model has already been trained. Loading file '{self.save_filename}'...")
            self.model.load_state_dict(torch.load(self.save_filename, weights_only=True))
            self.model.eval()
            return

        num_epochs = 100
        batch_size = 16
        learning_rate = 0.001
        max_steps = 100

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)

        mse_loss = nn.MSELoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}...")

            self.model.train()
            epoch_loss = 0.0

            for batch in range(10):  # 100 batches per epoch
                grids, target_counts = self.generate_training_set(batch_size)
                grids, target_counts = grids.to(device), target_counts.to(device)

                optimiser.zero_grad()
                predicted_counts, _ = self.model(grids, max_steps=max_steps)
                predicted_counts = predicted_counts.squeeze()

                # print(max(predicted_counts.tolist()))

                # print("Target:", target_counts.tolist())
                # print("Predicted:", predicted_counts.tolist())

                loss = mse_loss(predicted_counts, target_counts)

                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            # log epoch loss
            avg_epoch_loss = epoch_loss / 100
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        print("Training complete.")

        torch.save(self.model.state_dict(), self.save_filename)
        self.already_trained = True

    def demonstrate(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        max_steps = 100

        self.model.to(device)
        self.model.eval()

        grids, target_counts = self.generate_training_set(1)
        grids = grids.to(device)

        print(grids)
        print("Target count:", target_counts.tolist()[0])

        with torch.no_grad():
            predicted_counts, mask_memory = self.model(grids, max_steps=max_steps)
        print("Predicted count:", predicted_counts.tolist()[0])

        print("Mask memory:")
        print(mask_memory.view(self.grid_size, self.grid_size))
from abc import ABC, abstractmethod
from dataclasses import dataclass
from utilities.grid import GridFactory, GridBatch
from abstract_classes.trainer import Trainer
from models.models import GridAutoencoder, GridCounter
import os
import random
import torch
import torch.nn as nn

class GridAutoencoderTrainer(Trainer):
    def __init__(self, num_rows, num_cols):
        super().__init__("grid_autoencoder")
        self.num_rows = num_rows
        self.num_cols = num_cols

    @dataclass
    class TrainingPhase:
        training_type: str
        num_training_batches: str
        num_epochs: str

    def model_factory(self):
        return GridAutoencoder()

    def generate_training_set(self, type: str, num_batches: int):
        grid_factory = GridFactory(self.num_rows, self.num_cols)

        if type == "random":
            generator = grid_factory.generate_random
        elif type == "random_lines":
            generator = grid_factory.generate_random_line
        elif type == "random_lines_mixin_0.1":
            generator = lambda: grid_factory.generate_random_line(mixin_amount=0.1)

        return GridBatch.generate_batch(generator, num_batches)

    def train(
        self,
        training_phases: list,
        force_retrain=False
    ):
        if self.already_trained and force_retrain == False:
            print(f"This model has already been trained. Loading file '{self.save_filename}'...")
            self.model.load_state_dict(torch.load(self.save_filename, weights_only=True))
            self.model.eval()
            return

        for phase in training_phases:
            training_set = self.generate_training_set(
                phase.training_type,
                phase.num_training_batches
            )

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            for epoch in range(phase.num_epochs):
                for batch in torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True):
                    optimizer.zero_grad()
                    outputs = self.model(batch)
                    loss = loss_fn(outputs, batch)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        
        torch.save(self.model.state_dict(), self.save_filename)
        self.already_trained = True

    def demonstrate(self):
        self.model.eval()

        demo_example = self.generate_training_set("random_lines", 1)[0]
        
        with torch.no_grad():
            reconstructed = self.model(demo_example.unsqueeze(0)) # add batch dimension

        original_grid = demo_example.squeeze(0).squeeze(0).tolist()
        reconstructed_grid = reconstructed.squeeze(0).squeeze(0).tolist()

        print("Original Grid:")
        for row in original_grid:
            print(["{:.2f}".format(val) for val in row])

        print("\nReconstructed Grid:")
        for row in reconstructed_grid:
            print(["{:.2f}".format(val) for val in row])

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
        if force_retrain == False:
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

            for batch in range(10): # 100 batches per epoch
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
        
        print("Training complete.")

        torch.save(self.model.state_dict(), self.save_filename)
        self.already_trained = True

    def demonstrate(self):
        self.model.eval()

        max_steps = 100
        
        grids, target_counts = self.generate_training_set(1)
        print(grids)
        print("Target count:", target_counts.tolist()[0])

        predicted_counts, mask_memory = self.model(grids, max_steps=max_steps)
        print("Predicted count:", predicted_counts.tolist()[0])

        print("Mask memory:")
        print(mask_memory.view(self.grid_size, self.grid_size))

        
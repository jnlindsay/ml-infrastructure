from abc import ABC, abstractmethod
from dataclasses import dataclass
from grid import GridFactory, GridBatch
from models import GridAutoencoder, GridCounter
import os
import torch
import torch.nn as nn

class Trainer(ABC):
    def __init__(self, model_name: str):
        self.model = self.model_factory()
        self.model_name = model_name
        self.save_filename = model_name + ".pth"
        self.already_trained = os.path.exists(self.save_filename)

    @abstractmethod
    def model_factory():
        pass

    @abstractmethod
    def generate_training_set():
        pass
    
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def demonstrate():
        pass

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
        """
        !!! DUMMY DATASET !!!
        """
        grids = torch.randn((batch_size, 1, self.grid_size, self.grid_size))
        target_counts = torch.randint(1, 10, (batch_size, 1)).float()
        return grids, target_counts

    def train(self):
        num_epochs = 100
        batch_size = 16
        learning_rate = 0.001
        max_steps = 50

        # TODO: specify Mac device?
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        mse_loss = nn.MSELoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # training loop
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for _ in range(100): # 100 batches per epoch
                grids, target_counts = self.generate_training_set(batch_size)
                grids, target_counts = grids.to(device), target_counts.to(device)

                optimiser.zero_grad()
                predicted_counts, _ = self.model(grids, max_steps=max_steps)

                predicted_counts = predicted_counts.squeeze()
                loss = mse_loss(predicted_counts, target_counts)

                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            # log epoch loss
            avg_epoch_loss = epoch_loss / 100
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
        
        print("Training complete.")

    def demonstrate(self):
        model = GridCounter(
            self.input_size, 
            self.hidden_size,
            self.grid_size
        )

        grid = torch.randn((2, 1, self.grid_size, self.grid_size)) # batch size of 2

        final_count, final_mask = model(grid)
        print("Final Count:", final_count)
        print("Final Mask Memory:", final_mask.view(-1, self.grid_size, self.grid_size))
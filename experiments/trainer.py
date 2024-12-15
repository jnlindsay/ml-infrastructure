from abc import ABC, abstractmethod
from dataclasses import dataclass
from grid import GridFactory, GridBatch
from models import GridAutoencoder
import os
import torch
import torch.nn as nn

class Trainer(ABC):
    def __init__(self, modelClass, model_name: str):
        self.model = modelClass()
        self.model_name = model_name
        self.save_filename = model_name + ".pth"
        self.already_trained = os.path.exists(self.save_filename)

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
        super().__init__(GridAutoencoder, "grid_autoencoder")
        self.num_rows = num_rows
        self.num_cols = num_cols

    @dataclass
    class TrainingPhase:
        training_type: str
        num_training_batches: str
        num_epochs: str

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
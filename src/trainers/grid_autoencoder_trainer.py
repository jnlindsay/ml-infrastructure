from dataclasses import dataclass
from utilities.grid import GridFactory, GridBatch
from trainers.trainer import Trainer
from models.grid_autoencoder import GridAutoencoder
import torch
import torch.nn as nn
from utilities.hash import Hash

class GridAutoencoderTrainer(Trainer):
    def __init__(self, num_rows, num_cols):
        super().__init__("grid_autoencoder")
        self.num_rows = num_rows
        self.num_cols = num_cols

    @dataclass(frozen=True)
    class TrainingPhase:
        training_type: str
        num_training_batches: str
        num_epochs: str

    def model_factory(self):
        return GridAutoencoder()

    def generate_training_set(self, type: str, num_batches: int):
        grid_factory = GridFactory(self.num_rows, self.num_cols)
        generator = None

        if type == "random":
            generator = grid_factory.generate_random
        elif type == "random_symmetrical":
            generator = grid_factory.generate_random_symmetrical
        elif type == "random_lines":
            generator = grid_factory.generate_random_line
        elif type == "random_lines_mixin_0.1":
            generator = lambda: grid_factory.generate_random_line(mixin_amount=0.1)

        if generator is None: raise Exception("No generator specified")
        return GridBatch.generate_batch(generator, num_batches)

    def train(
        self,
        training_phases: list,
        force_retrain=False
    ):
        suffix = Hash.hash_to_string(training_phases)
        save_filepath = self.get_save_filepath(suffix=suffix)

        if self.save_file_exists(suffix=suffix) and force_retrain == False:
            print(f"Loading model from file '{save_filepath}'...")
            self.model.load_state_dict(torch.load(save_filepath, weights_only=True))
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
        
        torch.save(self.model.state_dict(), save_filepath)

    def demonstrate(self, type: str = None, demo_example=None):
        self.model.eval()

        if demo_example is None:
            if type == "random":
                pass
            elif type == "random_symmetrical":
                demo_example = self.generate_training_set("random_symmetrical", 1)[0]
            elif type == "random_lines":
                demo_example = self.generate_training_set("random_lines", 1)[0]
            elif type == "random_lines_mixin_0.1":
                pass

        if demo_example is None: raise Exception("No demo example specified")

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

        
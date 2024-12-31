from dataclasses import dataclass
from utilities.grid import GridFactory, GridBatch
from trainers.trainer import Trainer
from models.grid_autoencoder import GridAutoencoder, ViTAutoencoder, ViTSequenceAutoencoder
import torch
import torch.nn as nn
from utilities.hash import Hash
from utilities.visualiser import Visualiser


class GridSequenceAutoencoderTrainer(Trainer):
    def __init__(self, num_rows, num_cols):
        super().__init__("grid_autoencoder")
        self.num_rows = num_rows
        self.num_cols = num_cols

    @dataclass(frozen=True)
    class TrainingPhase:
        training_type: str
        num_training_batches: int
        num_epochs: int

    def model_factory(self):
        return ViTSequenceAutoencoder()

    def generate_training_set(self, type: str, batch_size: int):
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

        return GridBatch.generate_batch(generator, batch_size)

    def generate_sequence_training_set(self, type: str, batch_size: int, memory_length: int):
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

        batch = [
            torch.stack([generator().unsqueeze(0) for _ in range(memory_length + 1)])
            for _ in range(batch_size)
        ]

        return torch.stack(batch)  # (batch_size, memory_length + 1, 1, num_rows, num_cols)

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
            training_set = self.generate_sequence_training_set(
                phase.training_type,
                phase.num_training_batches,
                self.model.memory_length
            )

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            for epoch in range(phase.num_epochs):
                for batch in torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True):
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    outputs, current_frame = self.model(batch)
                    loss = loss_fn(outputs, current_frame)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        torch.save(self.model.state_dict(), save_filepath)

    def demonstrate(
            self,
            type: str = None,
            demo_example=None,
            show_loss=False,
    ):
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
            reconstructed = self.model(demo_example.unsqueeze(0))  # add batch dimension

        original_grid = demo_example.squeeze(0).squeeze(0)
        reconstructed_grid = reconstructed.squeeze(0).squeeze(0)

        print("Original Grid:")
        Visualiser.visualise(original_grid)

        print("\nReconstructed Grid:")
        Visualiser.visualise(reconstructed_grid)

        if show_loss:
            loss_fn = nn.MSELoss()
            print("Loss: {:.4f}".format(loss_fn(reconstructed_grid, original_grid).item()))
            print()
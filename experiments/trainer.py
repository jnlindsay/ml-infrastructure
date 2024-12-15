from abc import ABC, abstractmethod
from models import GridAutoencoder
from grid import GridFactory, GridBatch
import torch
import torch.nn as nn
import os

class Trainer(ABC):
    def __init__(self, modelClass, model_name: str):
        self.model = modelClass()
        self.save_filename = model_name + ".pth"
        self.already_trained = os.path.exists(self.save_filename)
    
    @abstractmethod
    def configure_training_set():
        pass
    
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def demonstrate():
        pass

class GridAutoencoderTrainer(Trainer):
    def __init__(self):
        super().__init__(GridAutoencoder, "grid_autoencoder")

    def configure_training_set(self, num_batches):
        grid_factory = GridFactory(10, 10)
        return GridBatch.generate_batch(
            lambda: grid_factory.generate_random_line(),
            num_batches
        )

    def train(self):
        if self.already_trained:
            self.model.load_state_dict(torch.load(self.save_filename, weights_only=True))
            self.model.eval()
            print("This model has already been trained.")
            print(f"The model from the file {self.save_filename} has been loaded.")
            return

        training_set = self.configure_training_set(1000)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(100):
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

        demo_example = self.configure_training_set(1)[0]
        
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
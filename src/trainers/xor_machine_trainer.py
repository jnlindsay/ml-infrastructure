from models.xor_machine import XorMachine
from trainers.trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim

class XorMachineTrainer(Trainer):
    def __init__(self, hidden_layer=False):
        self.hidden_layer = hidden_layer
        self.X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # inputs
        self.y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # outputs
        super().__init__("xor_machine")

    def model_factory(self):
        return XorMachine(hidden_layer=self.hidden_layer)

    def generate_training_set(self):
        pass

    def train(self):
        criterion = nn.MSELoss()
        num_epochs = 1000

        optimiser = optim.SGD(self.model.parameters(), lr=0.1)

        for epoch in range(num_epochs):
            outputs = self.model(self.X)
            loss = criterion(outputs, self.y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

        print()

    def demonstrate(self):
        with torch.no_grad():
            predictions = self.model(self.X)
            print("Predictions:")
            print(predictions)
            print()
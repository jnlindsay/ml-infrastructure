import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the dataset
class BlackCellDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.grid_size = 10

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random 10x10 grid with values 0 or 1
        grid = torch.randint(0, 2, (self.grid_size, self.grid_size), dtype=torch.float32)
        # Target is the count of black cells (1s)
        target = grid.sum()
        return grid.flatten(), target, grid

# Define the neural network
class FCN(nn.Module):
    def __init__(self, input_size):
        super(FCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
grid_size = 10
input_size = grid_size * grid_size
num_samples = 1000
batch_size = 32
epochs = 10
learning_rate = 0.001

# Create dataset and dataloaders
train_dataset = BlackCellDataset(num_samples)
test_dataset = BlackCellDataset(10)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)  # Batch size 1 for easy printing

# Initialize model, loss, and optimizer
model = FCN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    for inputs, targets, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Testing and printing
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets, grid in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item()
        
        # Print the grid
        print("\nGrid:")
        for row in grid.view(grid_size, grid_size):
            print("".join(["■" if cell == 1 else "□" for cell in row.int()]))
        
        # Print predicted and actual counts
        print(f"Actual count: {int(targets.item())}, Predicted count: {outputs.item():.2f}")

test_loss /= len(test_loader)
print(f"\nTest Loss: {test_loss:.4f}")
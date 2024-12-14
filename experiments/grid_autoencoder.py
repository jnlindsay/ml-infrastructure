import torch
import torch.nn as nn
import grid

class GridAutoencoder(nn.Module):
    def __init__(self):
        super(GridAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 10x10 -> 10x10
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 10x10 -> 10x10
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 32),  # Compress to 32-dim latent vector
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 16 * 10 * 10),  # Expand back
            nn.Unflatten(1, (16, 10, 10)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Reconstruct values between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Example data
grid_factory = grid.GridFactory(10, 10)
data = grid.GridBatch.generate_batch(
    lambda: grid_factory.generate_random_line(),
    1000
)

# Training loop
model = GridAutoencoder()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):  # Training for 20 epochs
    for batch in torch.utils.data.DataLoader(data, batch_size=32, shuffle=True):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

def demonstrate_autoencoder(model, data_sample):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Pass the sample through the model
        reconstructed = model(data_sample.unsqueeze(0))  # Add batch dimension

    # Convert to Python lists for easier printing
    original_grid = data_sample.squeeze(0).squeeze(0).tolist()
    reconstructed_grid = reconstructed.squeeze(0).squeeze(0).tolist()

    # Print the original and reconstructed grids
    print("Original Grid:")
    for row in original_grid:
        print(["{:.2f}".format(val) for val in row])

    print("\nReconstructed Grid:")
    for row in reconstructed_grid:
        print(["{:.2f}".format(val) for val in row])

# Select a random sample from the data to test the model
random_idx = torch.randint(0, len(data), (1,)).item()
data_sample = data[random_idx]

# Demonstrate the autoencoder's performance
demonstrate_autoencoder(model, data_sample)
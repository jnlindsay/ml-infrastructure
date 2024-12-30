import torch.nn as nn

class GridAutoencoder(nn.Module):
    def __init__(self):
        super(GridAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 10x10 -> 10x10
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 10x10 -> 10x10
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 32) # compress to 32-dim latent vector
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 16 * 10 * 10), # expand back
            nn.Unflatten(1, (16, 10, 10)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(), # reconstruct values between 0 and 1
        )

    def forward(self, x): # expects shape (batch_size, 1, 10, 10)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


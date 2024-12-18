import torch
import torch.nn as nn
import torch.nn.functional as F
import grid

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

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class GridCounter(nn.Module):
    def __init__(self, input_size, hidden_size, grid_size):
        super(GridCounter, self).__init__()
        self.hidden_size = hidden_size

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 10x10 -> 10x10
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 10x10 -> 10x10
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(grid_size * grid_size * 16, input_size) # compress to input_size
        )

        # GRU controller
        self.gru = nn.GRUCell(input_size + grid_size * grid_size + 1, hidden_size)

        # action heads
        self.write_head = nn.Linear(hidden_size, grid_size * grid_size)
        self.increment_head = nn.Linear(hidden_size, 1)
        self.terminate_head = nn.Linear(hidden_size, 1)

    def forward(self, grid, max_steps=100):
        """
        grid: Input grid (B, 1, H, W)
        max_steps: Max time steps to prevent infinite loops
        """

        batch_size, _, H, W = grid.shape
        device = grid.device

        # initialise states
        mask_memory = torch.zeros((batch_size, H * W), device=device)
        current_count = torch.zeros((batch_size, 1), device=device)
        hidden_state = torch.zeros((batch_size, self.hidden_size), device=device)
        cnn_features = self.cnn(grid)  # shape: (B, input_size)

        for t in range(max_steps):
            # concatenate inputs
            gru_input = torch.cat([cnn_features, mask_memory, current_count], dim=1)
            hidden_state = self.gru(gru_input, hidden_state)

            # predict actions
            write_memory = torch.sigmoid(self.write_head(hidden_state))
            increment = torch.sigmoid(self.increment_head(hidden_state))
            terminate = torch.sigmoid(self.terminate_head(hidden_state))

            # update states
            mask_memory = mask_memory + write_memory
            current_count = current_count + increment

            # check termination
            if (terminate > 0.5).all():
                break

        return current_count, mask_memory
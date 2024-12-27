import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

class GridEncoderStableBaselines(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space,
            pretrained_model,
            features_dim=128
    ):
        super(GridEncoderStableBaselines, self).__init__(
            observation_space,
            features_dim
        )

        self.pretrained_model = pretrained_model
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self._get_output_dim(observation_space), features_dim)

    def forward(self, observations):
        if observations.shape == (1, 4):
            observations = observations.view(1, 1, 2, 2)
        elif observations.shape == (64, 4):
            observations = observations.view(64, 1, 2, 2)

        with torch.no_grad():
            features = self.pretrained_model.encoder(observations)
        return self.fc(features)

    def _get_output_dim(self, observation_space):
        dummy_input = torch.zeros(1, *observation_space.shape)
        with torch.no_grad():
            features = self.pretrained_model.encoder(dummy_input.unsqueeze(0).unsqueeze(0))
        return features.shape[1]


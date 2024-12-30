import torch
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

class ViTAutoencoder(nn.Module):
    def __init__(self, patch_size=2, num_patches=25, hidden_dim=128, num_heads=4, num_layers=3):
        super(ViTAutoencoder, self).__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bottleneck
        self.latent_dim = 32
        self.to_latent = nn.Linear(hidden_dim * num_patches, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, hidden_dim * num_patches)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final reconstruction
        self.unpatch = nn.Sequential(
            nn.Linear(hidden_dim, patch_size * patch_size),
            nn.Unflatten(2, (patch_size, patch_size))
        )

    def forward(self, x): # (batch_size, 1, 10, 10)
        # Patch embedding
        x = self.patch_embed(x) # (batch_size, hidden_dim, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, hidden_dim)
        x = x + self.pos_embed

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Bottleneck
        batch_size = encoded.shape[0]
        flattened = encoded.reshape(batch_size, -1)
        latent = self.to_latent(flattened) # (batch_size, latent_dim)

        # Decode from latent
        decoded = self.from_latent(latent)
        decoded = decoded.reshape(batch_size, -1, self.hidden_dim)

        # Transformer decoding
        decoded = self.transformer_decoder(decoded, encoded)

        # Reshape to image
        patches = self.unpatch(decoded) # (batch_size, num_patches, patch_size, patch_size)

        # Reconstruct image
        height = width = 10
        patches = patches.unflatten(1, (height // self.patch_size, width // self.patch_size))
        output = patches.permute(0, 2, 4, 1, 3).reshape(batch_size, 1, height, width)

        return torch.sigmoid(output)


class ViTGridSequenceAutoencoder(nn.Module):
    def __init__(self, memory_length=5, patch_size=2, hidden_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.memory_length = memory_length
        num_patches = (10 // patch_size) * (10 // patch_size)  # 25 patches per frame
        total_patches = num_patches * memory_length  # 25 * memory_length total patches

        # Patch embedding (processes each frame independently)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        # Position embeddings for both spatial and temporal dimensions
        self.pos_embed = nn.Parameter(torch.randn(1, total_patches, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bottleneck
        self.latent_dim = 32
        self.to_latent = nn.Linear(hidden_dim * total_patches, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, hidden_dim * num_patches)  # Note: decode to single frame

        # Transformer decoder (decodes to single frame)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final reconstruction
        self.unpatch = nn.Sequential(
            nn.Linear(hidden_dim, patch_size * patch_size),
            nn.Unflatten(2, (patch_size, patch_size))
        )

    def forward(self, x):  # x shape: (batch_size, memory_length + 1, 1, 10, 10)
        batch_size = x.size(0)
        memory_frames = x[:, :self.memory_length]  # Exclude current frame
        current_frame = x[:, -1]  # Target frame

        # Process memory frames
        # Reshape to process all frames at once
        memory_frames = memory_frames.reshape(batch_size * self.memory_length, 1, 10, 10)

        # Patch embedding
        x = self.patch_embed(memory_frames)  # (batch*memory_length, hidden_dim, 25)
        x = x.transpose(1, 2)  # (batch*memory_length, 25, hidden_dim)

        # Reshape to combine all patches from all frames
        x = x.reshape(batch_size, -1, self.hidden_dim)  # (batch, memory_length*25, hidden_dim)
        x = x + self.pos_embed

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Bottleneck
        flattened = encoded.reshape(batch_size, -1)
        latent = self.to_latent(flattened)

        # Decode to single frame
        decoded = self.from_latent(latent)
        decoded = decoded.reshape(batch_size, -1, self.hidden_dim)  # (batch, 25, hidden_dim)

        # Transformer decoding (using encoded memory as context)
        decoded = self.transformer_decoder(decoded, encoded)

        # Reconstruct image
        patches = self.unpatch(decoded)  # (batch, 25, 2, 2)
        patches = patches.unflatten(1, (5, 5))
        output = patches.permute(0, 2, 4, 1, 3).reshape(batch_size, 1, 10, 10)

        return torch.sigmoid(output), current_frame  # Return prediction and target
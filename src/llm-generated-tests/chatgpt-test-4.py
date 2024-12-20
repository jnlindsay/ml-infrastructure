import torch
import torch.nn as nn
import torch.optim as optim

class NeuralGPU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_steps):
        """
        Args:
            input_dim: Number of input features (e.g., one-hot encoding size).
            hidden_dim: Number of hidden features (channels in the conv layer).
            kernel_size: Size of the convolutional kernel.
            num_steps: Number of recurrent steps to unroll.
        """
        super(NeuralGPU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps 

        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Recurrent convolutional layer
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # Output projection back to input dimension
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim).
        """
        batch_size, seq_len, input_dim = x.size()

        # Project input to hidden dimension
        h = self.input_proj(x)  # Shape: (batch_size, seq_len, hidden_dim)

        # Rearrange for convolution: (batch_size, channels, seq_len)
        h = h.permute(0, 2, 1)

        # Recurrent convolutional updates
        for _ in range(self.num_steps):
            h = self.conv(h)  # Apply convolution
            h = self.activation(h)  # Apply nonlinearity

        # Rearrange back: (batch_size, seq_len, hidden_dim)
        h = h.permute(0, 2, 1)

        # Project hidden states back to input dimension
        output = self.output_proj(h)

        return output

# Example task: binary addition
def generate_binary_addition_data(batch_size, seq_len):
    """
    Generate random binary addition tasks.
    Returns inputs (two binary numbers) and outputs (their sum).
    """
    x1 = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.float32)
    x2 = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.float32)

    # Stack inputs as two features
    inputs = torch.stack([x1, x2], dim=-1)  # Shape: (batch_size, seq_len, 2)

    # Compute sum and one-hot encode the result
    result = (x1 + x2).long()
    result_one_hot = torch.nn.functional.one_hot(result, num_classes=2).float()  # Shape: (batch_size, seq_len, 2)

    return inputs, result_one_hot

# Hyperparameters
input_dim = 2
hidden_dim = 16
kernel_size = 3
num_steps = 6
seq_len = 10
batch_size = 32
num_epochs = 100
learning_rate = 0.01

# Model, loss, optimizer
model = NeuralGPU(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_steps=num_steps)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Generate random data
    inputs, targets = generate_binary_addition_data(batch_size, seq_len)

    # Forward pass
    outputs = model(inputs)

    # Compute loss (cross-entropy over one-hot encoded targets)
    outputs_flat = outputs.view(-1, input_dim)
    targets_flat = targets.argmax(dim=-1).view(-1)  # Flatten to match shape
    loss = criterion(outputs_flat, targets_flat)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model on a new example
test_inputs, test_targets = generate_binary_addition_data(1, seq_len)
test_outputs = model(test_inputs).argmax(dim=-1)
print("Inputs:\n", test_inputs[0])
print("Predicted Outputs:\n", test_outputs[0])
print("Actual Outputs:\n", test_targets.argmax(dim=-1)[0])
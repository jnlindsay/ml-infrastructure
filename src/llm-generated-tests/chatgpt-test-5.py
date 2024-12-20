import torch
import torch.nn as nn
import torch.optim as optim

# Define the XOR dataset
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # Inputs
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # Targets

# Define the simple linear model: f(x) = x^T * w + b
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 input features, 1 output

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = SimpleModel()

# Define the Mean Squared Error (MSE) loss
criterion = nn.MSELoss()

# Use Stochastic Gradient Descent (SGD) for optimization
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: compute predicted y by passing x to the model
    outputs = model(X)
    
    # Compute the loss
    loss = criterion(outputs, y)
    
    # Backward pass: compute gradients
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# After training, test the model's predictions on the XOR inputs
with torch.no_grad():
    predictions = model(X)
    print("Predictions:")
    print(predictions)
import torch
import torch.nn as nn

class XorMachine(nn.Module):
    def __init__(self, hidden_layer = False):
        super().__init__()
        self.hidden = nn.Linear(2, 2) if hidden_layer else None
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        if self.hidden: x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
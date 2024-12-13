import numpy as np
import torch

class GridFactory():
    def __init__(
        self,
        num_rows,
        num_cols,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
    
    def generate_empty(self, default_val=0):
        return torch.empty(self.num_rows, self.num_cols).fill_(default_val)

    def generate_random(self, scale=1, offset=0):
        return torch.rand(self.num_rows, self.num_cols) * scale + offset

class GridBatch():
    def generate_batch(grid_generator, batch_size):
        grids = [grid_generator().unsqueeze(0) for _ in range(batch_size)]
        return torch.stack(grids)  # shape (batch_size, 1, num_rows, num_cols)

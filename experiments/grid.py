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

    def generate_random_line(self):
        grid = self.generate_empty()

        x_1 = np.random.randint(0, self.num_cols)
        x_2 = np.random.randint(0, self.num_rows)
        y_1 = np.random.randint(0, self.num_cols)
        y_2 = np.random.randint(0, self.num_rows)
        self.draw_bresenham_line(grid, x_1, x_2, y_1, y_2)

        return grid

    def draw_bresenham_line(self, grid, x_1, x_2, y_1, y_2):
        dx = abs(x_2 - x_1)
        dy = abs(y_2 - y_1)
        s_x = 1 if x_1 < x_2 else -1
        s_y = 1 if y_1 < y_2 else -1
        err = dx - dy

        while True:
            grid[y_1][x_1] = 1  # Mark the current cell
            if x_1 == x_2 and y_1 == y_2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x_1 += s_x
            if e2 < dx:
                err += dx
                y_1 += s_y

class GridBatch():
    def generate_batch(grid_generator, batch_size):
        grids = [grid_generator().unsqueeze(0) for _ in range(batch_size)]
        return torch.stack(grids)  # shape (batch_size, 1, num_rows, num_cols)

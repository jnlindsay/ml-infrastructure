import numpy as np
import torch
import random

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

    def generate_random_symmetrical(self, scale=1, offset=0):
        half_cols = (self.num_cols + 1) // 2
        left_half = torch.rand(self.num_rows, half_cols) * scale + offset
        left_half_flippable = left_half if self.num_cols % 2 == 0 else left_half[:, :-1]
        right_half = torch.flip(left_half_flippable, dims=[1])
        grid = torch.cat([left_half, right_half], dim=1)
        return grid

    def generate_random_spaced_dots(self, num_dots=2):
        grid = self.generate_empty()

        num_dots_remaining = num_dots
        while num_dots_remaining > 0:
            i = random.randint(0, self.num_cols - 1)
            j = random.randint(0, self.num_rows - 1)

            skip = False
            for ii in [i - 1, i, i + 1]:
                for jj in [j - 1, j, j + 1]:
                    if (
                        0 <= ii <= self.num_cols - 1 and
                        0 <= jj <= self.num_rows - 1 and
                        grid[ii][jj] == 1.0
                    ):
                        skip = True
            
            if skip is False:
                grid[i][j] = 1.0
                num_dots_remaining -= 1

        return grid

    def generate_random_line(self, mixin_amount=0):
        grid = self.generate_empty()

        x_1 = np.random.randint(0, self.num_cols)
        x_2 = np.random.randint(0, self.num_rows)
        y_1 = np.random.randint(0, self.num_cols)
        y_2 = np.random.randint(0, self.num_rows)
        self.draw_bresenham_line(grid, x_1, x_2, y_1, y_2)

        if mixin_amount > 0:
            grid = self.mixin_random(grid, mixin_amount)

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
    
    def mixin_random(self, input_grid, amount: float):
        output_grid = input_grid + amount * self.generate_random()
        return output_grid.clamp(0, 1)

    def generate_random_embedded_square_even(self, square_size: int):
        if self.num_cols != self.num_rows:
            raise ValueError("Grid must be a square")

        grid_size = self.num_cols
        assert grid_size == self.num_rows

        if grid_size % 2 != 0:
            raise ValueError("Grid size must be even")
        if square_size % 2 != 0:
            raise ValueError("Embedded square size must be even")
        if square_size > grid_size:
            raise ValueError("Square size cannot be greater than grid size")

        grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        start_idx = (self.num_rows - square_size) // 2
        end_idx = start_idx + square_size

        grid[start_idx:end_idx, start_idx:end_idx] = np.random.randint(0, 2, (square_size, square_size), dtype=np.int8)

        return grid

class GridBatch():
    def generate_batch(grid_generator, batch_size):
        grids = [grid_generator().unsqueeze(0) for _ in range(batch_size)]
        return torch.stack(grids)  # shape (batch_size, 1, num_rows, num_cols)

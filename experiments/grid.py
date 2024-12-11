import numpy as np

class Grid():
    def __init__(self, num_rows, num_cols, default_val=0):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid = np.full((self.num_rows, self.num_cols), default_val)

    def print(self):
        print()
        for row in self.grid: print(row)
        print()

    def random(self, values: list):
        self.grid = np.random.choice(values, size=(self.num_rows, self.num_cols))
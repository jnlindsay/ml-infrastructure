import torch
import matplotlib.pyplot as plt

class Visualiser:
    @staticmethod
    def visualise(grid):
        if len(grid.shape) != 2:
            raise Exception("Grid must be 2-dimensional")

        if type(grid) == torch.Tensor:
            grid = grid.numpy()

        plt.imshow(grid, cmap="gray_r", interpolation="none")
        plt.title("Grid visualisation")
        plt.show()
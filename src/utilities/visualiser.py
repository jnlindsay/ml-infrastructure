import torch
import matplotlib.pyplot as plt

class Visualiser:
    @staticmethod
    def visualise(grid: torch.Tensor):
        if len(grid.shape) != 2:
            raise Exception("Grid must be 2-dimensional")

        plt.imshow(grid.numpy(), cmap="gray", interpolation="none")
        plt.title("Grid visualisation")
        plt.show()
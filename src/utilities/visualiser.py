import torch
import matplotlib.pyplot as plt

class Visualiser:
    @staticmethod
    def visualise(grid, figsize=(3, 3)):
        if len(grid.shape) != 2:
            raise Exception("Grid must be 2-dimensional")

        if type(grid) == torch.Tensor:
            grid = grid.numpy()

        plt.figure(figsize=figsize)
        plt.imshow(grid, cmap="gray_r", interpolation="none")
        plt.xticks([])
        plt.yticks([])
        plt.show()
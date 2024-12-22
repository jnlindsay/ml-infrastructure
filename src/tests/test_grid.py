import pytest
import torch
from utilities.grid import GridFactory

NUM_ROWS = 10
NUM_COLS = 12

@pytest.fixture(scope="class")
def grid():
    return GridFactory(NUM_ROWS, NUM_COLS)

@pytest.fixture(scope="class")
def grid_even_width():
    return GridFactory(NUM_ROWS, 6)

@pytest.fixture(scope="class")
def grid_odd_width():
    return GridFactory(NUM_ROWS, 5)

class TestGrid:
    def test_generate_empty(self, grid):
        result = grid.generate_empty()
        assert result.shape == torch.Size([NUM_ROWS, NUM_COLS])

    def test_generate_random(self, grid):
        result = grid.generate_random()
        assert result.shape == torch.Size([NUM_ROWS, NUM_COLS])

    def test_generate_random_symmetrical(self, grid_even_width, grid_odd_width):
        results = [
            grid_odd_width.generate_random_symmetrical(),
            grid_even_width.generate_random_symmetrical()
        ]

        for result in results:
            height, width = result.shape
            for i in range(height):
                for j in range((width + 1) // 2):
                    assert result[i, j] == result[i, width - j - 1]
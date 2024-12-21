import pytest
import torch
from utilities.grid import GridFactory

NUM_ROWS = 10
NUM_COLS = 12

@pytest.fixture(scope="class")
def grid():
    grid = GridFactory(NUM_ROWS, NUM_COLS)
    return grid

class TestGrid:
    def test_generate_empty(self, grid):
        result = grid.generate_empty()
        assert result.shape == torch.Size([NUM_ROWS, NUM_COLS])

    def test_generate_random(self, grid):
        result = grid.generate_random()
        assert result.shape == torch.Size([NUM_ROWS, NUM_COLS])
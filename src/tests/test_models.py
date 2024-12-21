import pytest
from models.grid_counter import GridCounter
from models.grid_autoencoder import GridAutoencoder

class TestModels:
    def test_model_instantiations(self):
        _ = GridCounter(1, 2, 3)
        _ = GridAutoencoder()
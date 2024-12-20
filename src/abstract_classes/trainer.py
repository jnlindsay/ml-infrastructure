from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random

class Trainer(ABC):
    def __init__(self, model_name: str):
        self.model = self.model_factory()
        self.model_name = model_name
        self.save_filename = model_name + ".pth"
        self.already_trained = os.path.exists(self.save_filename)

    @abstractmethod
    def model_factory():
        pass

    @abstractmethod
    def generate_training_set():
        pass
    
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def demonstrate():
        pass
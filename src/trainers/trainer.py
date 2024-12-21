from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random

class Trainer(ABC):
    def __init__(self, model_name: str):
        self.model = self.model_factory()
        self.model_name = model_name
        self.save_filename = model_name + ".pth"

    @abstractmethod
    def model_factory(self):
        pass

    @abstractmethod
    def generate_training_set(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def demonstrate(self):
        pass

    def save_file_exists(self) -> bool:
        return os.path.exists(self.save_filename)
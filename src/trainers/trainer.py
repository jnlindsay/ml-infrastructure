from abc import ABC, abstractmethod
import os

class Trainer(ABC):
    def __init__(self, model_name: str):
        self.model = self.model_factory()
        self.model_name = model_name

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

    def get_save_filepath(self, suffix: str = None):
        class_dir = os.path.dirname(os.path.abspath(__file__))
        saved_models_dir = os.path.join(class_dir, "saved_models")
        os.makedirs(saved_models_dir, exist_ok=True)
        return os.path.join(saved_models_dir, self.model_name + ("_" + suffix if suffix else "") + ".pth")

    def save_file_exists(self, suffix: str = None) -> bool:
        return os.path.exists(self.get_save_filepath(suffix=suffix))
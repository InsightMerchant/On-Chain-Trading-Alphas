import importlib
import os
from utils.interfaces import ModelInterface

class ModelLoader:
    def __init__(self, models_package="strategies.models"):
        self.models_package = models_package

    def discover_models(self):
        models = []
        base_path = os.path.join(os.getcwd(), *self.models_package.split('.'))
        for filename in os.listdir(base_path):
            if filename.endswith(".py") and filename != "__init__.py":
                models.append(filename[:-3])
        return models

    def load_model(self, model_name):
        try:
            module = importlib.import_module(f"{self.models_package}.{model_name}")
            model_class = None
            for attr in dir(module):
                candidate = getattr(module, attr)
                if (isinstance(candidate, type) and 
                    issubclass(candidate, ModelInterface) and 
                    candidate is not ModelInterface):
                    model_class = candidate
                    break
            if model_class is None:
                raise ImportError(f"Module '{model_name}' does not implement a valid model class.")
            instance = model_class()
            if not hasattr(instance, "calculate_signal"):
                raise ImportError(f"Model in '{model_name}' lacks 'calculate_signal'.")
            config = getattr(model_class, "MODEL_CONFIG", {})
            return instance.calculate_signal, config
        except ImportError as e:
            print(f"Error loading model '{model_name}': {e}")
            return None, {}

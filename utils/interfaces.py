# utils/interfaces.py
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    MODEL_CONFIG = {}
    
    @abstractmethod
    def calculate_signal(self, df, **params):
        pass

class LogicInterface(ABC):
    @abstractmethod
    def apply(self, signal, buy_threshold, sell_threshold, position):
        pass

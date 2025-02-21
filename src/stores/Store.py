from abc import ABC, abstractmethod

class Store(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def ingest(self, data):
        pass

from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self, model:str):
        pass

    @abstractmethod
    def invoke(self, data):
        pass
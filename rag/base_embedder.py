from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()

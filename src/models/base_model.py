from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Abstract base class for constituency parsing models.
    """
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, sentence: str) -> Any:
        """
        Predicts the constituency tree/tags for a given sentence.
        Returns a structured representation (e.g., NLTK Tree or list of tokens with tags).
        """
        pass


import nltk
from .base_model import BaseModel

class DummyModel(BaseModel):
    """
    A dummy model that returns a fixed tree or random noise for testing.
    """
    def __init__(self, name: str, variation: bool = False):
        super().__init__(name)
        self.variation = variation

    def predict(self, sentence: str):
        # Basic NLTK tokenization and dummy tagging
        tokens = nltk.word_tokenize(sentence)
        if self.variation and "today" in tokens:
            # Simulate the "false positive" mentioned in user story
            # "today" might be NN in one model
            tagged = [(word, 'NN' if word == 'today' else 'DT') for word in tokens]
        else:
             # Correct one: 'today' as RB (Adverb) - simplified example
            tagged = [(word, 'RB' if word == 'today' else 'DT') for word in tokens]
            
        return tagged

    def get_tree_string(self, sentence: str):
        """
        Returns a dummy tree string for visualization testing.
        """
        # Simple flat tree structure for testing
        tokens = nltk.word_tokenize(sentence)
        tree_str = "(S " + " ".join([f"(NP {t})" for t in tokens]) + ")"
        return tree_str

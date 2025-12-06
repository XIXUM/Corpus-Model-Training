import os
from typing import Any, Optional
from .base_model import BaseModel

class SuparWrapper(BaseModel):
    """
    Wrapper for SuPar's CRF Constituency Parser (supports BERT).
    """
    def __init__(self, name: str, model_name: str = "crf-con-bert-en"):
        super().__init__(name)
        self.model_name = model_name
        
        print(f"Initializing SuPar ({model_name})...")
        try:
            from supar import Parser
            # SuPar downloads the model automatically if not present
            self.parser = Parser.load(model_name)
        except ImportError:
            print("Error: 'supar' library not found. Please install it with `pip install supar`.")
            raise
        except Exception as e:
            print(f"Error loading SuPar model '{model_name}': {e}")
            raise

    def predict(self, sentence: str) -> Any:
        """
        Predicts POS tags (extracted from the tree).
        Returns a list of (token, tag) tuples.
        """
        # SuPar predict returns a Dataset object holding the result
        # We pass prob=True or False? Default is sufficient.
        # SuPar expects tokenized input or raw string. Raw string is easier.
        try:
            # supar.predict handles list of sentences or single string.
            # It returns a result object.
            dataset = self.parser.predict(sentence, verbose=False, lang='en')
            # The result contains trees.
            # dataset[0] is the result for the first sentence.
            tree = dataset[0] # This is usually an nltk.Tree or SuPar's tree wrapper that converts to it
            return tree.pos()
        except Exception as e:
            print(f"Error in SuPar prediction: {e}")
            return []

    def get_tree_string(self, sentence: str) -> Optional[str]:
        """
        Returns the parse tree string (PTB format).
        """
        try:
            dataset = self.parser.predict(sentence, verbose=False, lang='en')
            if len(dataset) > 0:
                # dataset[0] is the tree object, str() converts it to bracketed string
                return str(dataset[0])
        except Exception as e:
            print(f"Error getting SuPar tree: {e}")
        return None


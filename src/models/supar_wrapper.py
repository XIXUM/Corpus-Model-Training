import os
from typing import Any, Optional
from .base_model import BaseModel

class SuparWrapper(BaseModel):
    """
    Wrapper for SuPar's CRF Constituency Parser (supports BERT).
    """
    def __init__(self, name: str, model_name: str = "crf-con-roberta-en"):
        super().__init__(name)
        
        # Map simplified user-friendly names to actual SuPar model keys
        # Based on inspection: 'crf-con-roberta-en' is available. 
        # 'crf-con-bert-en' was not found, defaulting to RoBERTa (a robust BERT variant)
        
        # If user asked for "bert" (via main.py mapping), we now map to roberta
        if model_name == "crf-con-bert-en":
            print("Notice: 'crf-con-bert-en' key not found in SuPar registry. Switching to 'crf-con-roberta-en'.")
            self.model_name = "crf-con-roberta-en"
        else:
            self.model_name = model_name
        
        print(f"Initializing SuPar ({self.model_name})...")
        try:
            from supar import Parser
            self.parser = Parser.load(self.model_name)
            
        except ImportError:
            print("Error: 'supar' library not found. Please install it with `pip install supar`.")
            raise
        except ValueError as e:
            if "unknown url type" in str(e):
                print(f"Error loading SuPar model: {e}")
                print(f"Model '{self.model_name}' might be invalid. Available keys usually include 'crf-con-en' (LSTM), 'crf-con-roberta-en'.")
                raise
            raise
        except Exception as e:
            print(f"Error loading SuPar model '{self.model_name}': {e}")
            raise

    def predict(self, sentence: str) -> Any:
        """
        Predicts POS tags (extracted from the tree).
        Returns a list of (token, tag) tuples.
        """
        try:
            # supar.predict handles list of sentences or single string.
            # It returns a result object.
            dataset = self.parser.predict(sentence, verbose=False, lang='en')
            # dataset[0] is the result for the first sentence.
            # In SuPar 1.1.4, accessing the tree might need specific method if it's not subscriptable?
            # dataset usually is iterable.
            if dataset and len(dataset) > 0:
                 tree = dataset[0] 
                 return tree.pos()
            return []
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
                return str(dataset[0])
        except Exception as e:
            print(f"Error getting SuPar tree: {e}")
        return None

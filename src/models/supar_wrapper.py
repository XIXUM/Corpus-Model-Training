import os
import sys
import torch
import supar
import supar.utils.config
import supar.models.const
import supar.utils.transform
import supar.utils.tokenizer
from supar import Parser
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
        
        # Move imports to top-level if possible, but keep workarounds local to avoid polluting global namespace unnecessarily 
        # unless strictly required. However, sys.modules injection MUST be done before torch.load.
        
        print(f"Initializing SuPar ({self.model_name})...")
        try:
            
            # WORKAROUND 1: Missing module 'supar.config'
            if 'supar.config' not in sys.modules:
                sys.modules['supar.config'] = supar.utils.config

            # WORKAROUND 2: Missing module 'supar.models.const.crf'
            if 'supar.models.const.crf' not in sys.modules:
                sys.modules['supar.models.const.crf'] = supar.models.const
            
            # WORKAROUND 3: Missing module 'supar.models.const.crf.transform'
            if 'supar.models.const.crf.transform' not in sys.modules:
                sys.modules['supar.models.const.crf.transform'] = supar.utils.transform

            # WORKAROUND 4: Missing attribute 'TransformerTokenizer'
            if not hasattr(supar.utils.tokenizer, 'TransformerTokenizer'):
                supar.utils.tokenizer.TransformerTokenizer = supar.utils.tokenizer.Tokenizer

            # WORKAROUND 5: PyTorch 2.6+ defaults to weights_only=True
            # which breaks supar's loading. We temporarily patch torch.load.
            original_load = torch.load
            
            def unsafe_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
                
            try:
                print("Applying temporary patch to torch.load for SuPar initialization...")
                torch.load = unsafe_load
                self.parser = Parser.load(self.model_name)
            finally:
                torch.load = original_load
            
            # CRITICAL: DO NOT REMOVE THIS TOKENIZER REPAIR BLOCK
            # The unpickled Tokenizer/TransformerTokenizer is incompatible with the current environment
            # (likely due to missing 'pipeline' attribute or Stanza initialization issues).
            # We aggressively replace it with a fresh, working instance.
            print("Replacing SuPar tokenizer with a fresh instance...")
            try:
                # Initialize a fresh tokenizer (uses Stanza under the hood)
                # We catch potential errors during initialization (e.g. download issues)
                fresh_tokenizer = supar.utils.tokenizer.Tokenizer(lang='en')
                
                if hasattr(self.parser, 'tokenizer'):
                    self.parser.tokenizer = fresh_tokenizer
                    print("Replaced self.parser.tokenizer")
                
                if hasattr(self.parser, 'transform') and hasattr(self.parser.transform, 'tokenizer'):
                    self.parser.transform.tokenizer = fresh_tokenizer
                    print("Replaced self.parser.transform.tokenizer")
                    
            except Exception as e:
                print(f"Warning: Failed to replace tokenizer: {e}")
            # END OF CRITICAL BLOCK
                
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
            # We wrap in list to prevent supar from treating the string as a file path 
            # (e.g. if sentence is "." which exists as a directory).
            dataset = self.parser.predict([sentence], verbose=False, lang='en')
            
            # dataset[0] is the result for the first sentence.
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
            # Wrap in list to avoid file path ambiguity
            dataset = self.parser.predict([sentence], verbose=False, lang='en')
            if len(dataset) > 0:
                return str(dataset[0])
        except Exception as e:
            print(f"Error getting SuPar tree: {e}")
        return None

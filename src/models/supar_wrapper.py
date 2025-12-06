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
            # The unpickled Tokenizer/TransformerTokenizer is incompatible/broken.
            # Since we perform manual tokenization using NLTK in predict(), we explicitly 
            # DISABLE the internal tokenizer in all fields to prevent double-tokenization 
            # and avoid the 'pipeline' attribute error.
            print("Disabling SuPar internal tokenizer in fields (using external NLTK tokenization)...")
            try:
                from collections.abc import Iterable
                
                # Disable in main tokenizer attributes if they exist
                if hasattr(self.parser, 'tokenizer'):
                    self.parser.tokenizer = None
                
                if hasattr(self.parser, 'transform'):
                    if hasattr(self.parser.transform, 'tokenizer'):
                        self.parser.transform.tokenizer = None
                        
                    # Crucial: Disable tokenizer in all data fields (WORD, POS, etc.)
                    # Note: Fields can be single objects or lists of objects
                    if hasattr(self.parser.transform, 'fields'):
                        for field_name in self.parser.transform.fields:
                            if hasattr(self.parser.transform, field_name):
                                field_val = getattr(self.parser.transform, field_name)
                                
                                # Handle list of fields
                                if isinstance(field_val, Iterable) and not isinstance(field_val, str):
                                    fields_to_check = field_val
                                else:
                                    fields_to_check = [field_val]
                                    
                                for f in fields_to_check:
                                    if hasattr(f, 'tokenize'):
                                        f.tokenize = None
                                        print(f"Disabled tokenizer in field '{field_name}'")
                    
                    # Also iterate over flattened_fields directly to be absolutely sure we caught everything
                    # (Parser.predict uses flattened_fields)
                    if hasattr(self.parser.transform, 'flattened_fields'):
                        for i, f in enumerate(self.parser.transform.flattened_fields):
                            if hasattr(f, 'tokenize') and f.tokenize is not None:
                                f.tokenize = None
                                print(f"Disabled tokenizer in flattened field {i} (name: {getattr(f, 'name', 'Unknown')})")
                    
            except Exception as e:
                print(f"Warning: Failed to disable tokenizer: {e}")
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
            # Tokenize manually to avoid SuPar's internal Tokenizer issues
            import nltk
            tokens = nltk.word_tokenize(sentence)
            
            # supar.predict handles list of sentences.
            # We pass a list of tokens (nested list) and lang=None to bypass internal tokenization
            dataset = self.parser.predict([tokens], verbose=False, lang=None)
            
            # dataset[0] is the result for the first sentence.
            if dataset and len(dataset) > 0:
                 su_sentence = dataset[0]
                 # SuPar Sentence object doesn't have .pos() method directly.
                 # We need to access the underlying NLTK tree from the 'TREE' field.
                 if hasattr(su_sentence, 'TREE'):
                     return su_sentence.TREE.pos()
                 elif hasattr(su_sentence, 'values') and len(su_sentence.values) > 2:
                     # Fallback: TreeSentence values are [words, tags, tree, chart]
                     return su_sentence.values[2].pos()
                 
                 # Fallback 2: If str() works and returns bracketed string, parse it
                 try:
                     return nltk.Tree.fromstring(str(su_sentence)).pos()
                 except:
                     pass
                     
                 return []
            return []
        except Exception as e:
            print(f"Error in SuPar prediction: {e}")
            return []

    def get_tree_string(self, sentence: str) -> Optional[str]:
        """
        Returns the parse tree string (PTB format).
        """
        try:
            # Tokenize manually
            import nltk
            tokens = nltk.word_tokenize(sentence)
            
            # Pass list of tokens and lang=None
            dataset = self.parser.predict([tokens], verbose=False, lang=None)
            
            if len(dataset) > 0:
                su_sentence = dataset[0]
                # Prefer accessing TREE field directly
                if hasattr(su_sentence, 'TREE'):
                    return su_sentence.TREE.pformat()
                return str(su_sentence)
        except Exception as e:
            print(f"Error getting SuPar tree: {e}")
        return None

import spacy
import benepar
import sys
from typing import Any, List, Tuple
from spacy.tokens import Doc
from .base_model import BaseModel

# Define pipe name constant
PIPE_BENE_PAR = 'benepar'

class BeneparWrapper(BaseModel):
    """
    Wrapper for Benepar model using spaCy pipeline.
    Includes safe error handling and fallback mechanisms.
    """
    
    def __init__(self, name: str, model_name: str = "benepar_en3"):
        super().__init__(name)
        self.model_name = model_name
        
        # Initialize spaCy
        # We use a blank model and add benepar, or load a small model if available.
        # For this environment, we might default to blank 'en' + benepar.
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: en_core_web_sm not found. Using blank 'en' model.")
            self.nlp = spacy.blank('en')

        # Download benepar model if needed (this might fail in sandbox without network)
        # Assuming it's handled or we catch the error.
        try:
            benepar.download(model_name)
        except Exception as e:
            print(f"Warning: Could not download benepar model '{model_name}': {e}")

        # Add benepar to pipeline
        if PIPE_BENE_PAR not in self.nlp.pipe_names:
            try:
                self.nlp.add_pipe(PIPE_BENE_PAR, config={"model": model_name})
            except Exception as e:
                print(f"Error adding benepar pipe: {e}")

        # Apply the safe wrapper logic requested by user
        self._apply_safe_wrapper()

    def _apply_safe_wrapper(self):
        """
        Applies the user-provided safe wrapper logic to the benepar component.
        """
        if PIPE_BENE_PAR in self.nlp.pipe_names:
            
            # Define fallback parser (simple identity or dummy)
            def fallback_parser(doc):
                # Minimal fallback: just set the sentence structure to flat S -> [words]
                # or simply return doc without parse info
                print(f"Fallback parser triggered for: {doc.text[:20]}...")
                return doc

            # Define safe wrapper
            def safe_benepar_parser(doc):
                try:
                    # Try to use the benepar component if it exists
                    # Note: nlp.get_pipe returns the component object.
                    # In spaCy 3.x, components are callables.
                    # However, we are REPLACING the pipe in the pipeline, 
                    # so we can't call the pipe we are currently 'inside' or replaced?
                    # The user's code: return nlp.get_pipe(pipe_BENE_PAR)(doc)
                    # If we replace the pipe, nlp.get_pipe(pipe_BENE_PAR) returns THIS function!
                    # Recursive loop risk!
                    
                    # The user's intention is likely: 
                    # Wrap the *original* benepar component function.
                    
                    # Get the original component instance
                    # But we can't access it if we replaced it in the pipeline object.
                    # So we must capture it via closure BEFORE replacing.
                    pass # See below implementation
                    
                except Exception as e:
                    pass
            
            # Correct way to wrap:
            original_benepar = self.nlp.get_pipe(PIPE_BENE_PAR)
            
            def safe_benepar_parser_impl(doc):
                try:
                    return original_benepar(doc)
                except StopIteration as e:
                    print(f"❌ StopIteration error in benepar parsing!")
                    print(f"Problematic text: '{doc.text}'")
                    print(f"Text length: {len(doc.text)}")
                    print(f"Tokens: {[token.text for token in doc]}")
                    print(f"Error details: {e}")
                    print("Using fallback parser instead...")
                    return fallback_parser(doc)
                except Exception as e:
                    print(f"❌ Error in benepar parsing!")
                    print(f"Problematic text: '{doc.text}'")
                    print(f"Error details: {e}")
                    print("Using fallback parser instead...")
                    return fallback_parser(doc)

            try:
                # Replace the pipe
                self.nlp.replace_pipe(PIPE_BENE_PAR, safe_benepar_parser_impl)
                print("✓ Benepar component wrapped with error handling")
            except Exception as e:
                print(f"Warning: Could not wrap benepar component: {e}")

    def predict(self, sentence: str) -> Any:
        """
        Predicts the constituency tree and POS tags.
        """
        doc = self.nlp(sentence)
        
        # Extract POS tags
        # Result format: List of (token, tag) tuples to match DummyModel
        pos_tags = [(token.text, token.tag_) for token in doc]
        
        # Try to extract tree if available
        if len(list(doc.sents)) > 0:
            sent = list(doc.sents)[0]
            try:
                # Check if parse_string is available
                if hasattr(sent._, 'parse_string') and sent._.parse_string:
                    # It is available
                    self.last_tree_str = sent._.parse_string
                else:
                    self.last_tree_str = None
            except Exception:
                 self.last_tree_str = None
        
        return pos_tags

    def display_tree(self, sentence: str):
        """
        Parses and displays the tree structure.
        """
        doc = self.nlp(sentence)
        if len(list(doc.sents)) > 0:
            sent = list(doc.sents)[0]
            try:
                print(f"Tree for: {sentence[:50]}...")
                # _ is the extension attribute space
                # Benepar adds 'parse_string'
                if hasattr(sent._, 'parse_string') and sent._.parse_string:
                     # Convert to NLTK tree for display
                    from nltk import Tree
                    tree = Tree.fromstring(sent._.parse_string)
                    tree.pretty_print()
                else:
                    print("(No parse tree available)")
            except Exception as e:
                print(f"Error displaying tree: {e}")



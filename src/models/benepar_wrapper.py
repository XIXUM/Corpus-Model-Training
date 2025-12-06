import spacy
import benepar
import sys
import os
from typing import Any, List, Tuple, Optional
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
        self.last_tree_str = None
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: en_core_web_sm not found. Using blank 'en' model.")
            self.nlp = spacy.blank('en')

        # Check if model_name is a local path or a download name
        is_local_path = os.path.exists(model_name) or os.path.isdir(model_name)

        if not is_local_path:
            # Download benepar model if needed
            try:
                benepar.download(model_name)
            except Exception as e:
                # If it's not a known model name, it might be a path that doesn't exist yet or user error
                # But benepar.download is chatty.
                print(f"Warning: Could not download benepar model '{model_name}': {e}")
        else:
            print(f"Using local Benepar model from: {model_name}")

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
            
            def fallback_parser(doc):
                print(f"Fallback parser triggered for: {doc.text[:20]}...")
                return doc

            original_benepar = self.nlp.get_pipe(PIPE_BENE_PAR)
            
            def safe_benepar_parser_impl(doc):
                try:
                    return original_benepar(doc)
                except StopIteration as e:
                    print(f"❌ StopIteration error in benepar parsing!")
                    print(f"Problematic text: '{doc.text}'")
                    print(f"Text length: {len(doc.text)}")
                    print(f"Error details: {e}")
                    return fallback_parser(doc)
                except Exception as e:
                    print(f"❌ Error in benepar parsing!")
                    print(f"Problematic text: '{doc.text}'")
                    print(f"Error details: {e}")
                    return fallback_parser(doc)
            
            try:
                self.nlp.replace_pipe(PIPE_BENE_PAR, safe_benepar_parser_impl)
                print("✓ Benepar component wrapped with error handling")
            except Exception as e:
                print(f"Warning: Could not wrap benepar component: {e}")

    def predict(self, sentence: str) -> Any:
        """
        Predicts the constituency tree and POS tags.
        """
        doc = self.nlp(sentence)
        
        pos_tags = [(token.text, token.tag_) for token in doc]
        
        # Store tree string for later retrieval to avoid re-parsing
        if len(list(doc.sents)) > 0:
            sent = list(doc.sents)[0]
            try:
                if hasattr(sent._, 'parse_string') and sent._.parse_string:
                    self.last_tree_str = sent._.parse_string
                else:
                    self.last_tree_str = None
            except Exception:
                 self.last_tree_str = None
        
        return pos_tags

    def get_tree_string(self, sentence: str) -> Optional[str]:
        """
        Returns the parse string for the sentence. 
        """
        # If we just predicted this sentence, use cached result
        # A simple check if it matches the last one is tricky without storing the input sentence.
        # For safety, let's just re-parse or trust the caller knows.
        # Or better: just run nlp again. It's safer.
        doc = self.nlp(sentence)
        if len(list(doc.sents)) > 0:
            sent = list(doc.sents)[0]
            try:
                if hasattr(sent._, 'parse_string') and sent._.parse_string:
                    return sent._.parse_string
            except:
                pass
        return None

    def display_tree(self, sentence: str):
        """
        Parses and displays the tree structure.
        """
        tree_str = self.get_tree_string(sentence)
        if tree_str:
            try:
                print(f"Tree for: {sentence[:50]}...")
                from nltk import Tree
                tree = Tree.fromstring(tree_str)
                tree.pretty_print()
            except Exception as e:
                print(f"Error displaying tree: {e}")
        else:
            print(f"(No parse tree available for: {sentence[:30]}...)")

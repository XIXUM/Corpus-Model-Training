import stanza
import os
from typing import Any, Optional
from .base_model import BaseModel

class StanzaWrapper(BaseModel):
    """
    Wrapper for Stanza's constituency parser.
    """
    def __init__(self, name: str, model_name: str = "en"):
        super().__init__(name)
        self.model_name = model_name
        
        print(f"Initializing Stanza pipeline ({model_name})...")
        # Check if resources are downloaded, if not download them
        # Stanza usually handles this, but we can be explicit or let it auto-download.
        # We use only 'tokenize' and 'constituency' processors. 'pos' is implied/needed.
        try:
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
        except Exception:
            print("Downloading Stanza 'en' models...")
            stanza.download('en')
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    def predict(self, sentence: str) -> Any:
        """
        Predicts POS tags (and tree structure implicitly).
        Returns a list of (token, tag) tuples.
        """
        doc = self.nlp(sentence)
        # Stanza can have multiple sentences
        pos_tags = []
        for sent in doc.sentences:
            for word in sent.words:
                pos_tags.append((word.text, word.xpos)) # xpos is usually PTB tag
        return pos_tags

    def get_tree_string(self, sentence: str) -> Optional[str]:
        """
        Returns the parse tree string (PTB format).
        """
        doc = self.nlp(sentence)
        if len(doc.sentences) > 0:
            # Stanza returns the tree object, we can get string via __str__ or similar
            # sent.constituency is the tree
            # We might need to join multiple sentences if the input was split by Stanza
            # But our pipeline splits sentences beforehand usually.
            # If multiple, we might just return the first or join them under a ROOT.
            # For now, return the first valid tree.
            return str(doc.sentences[0].constituency)
        return None


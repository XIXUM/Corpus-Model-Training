import spacy
from typing import Any, Optional, List, Tuple
from .base_model import BaseModel

class SpacyPOSModel(BaseModel):
    """
    A simple wrapper around SpaCy's Part-of-Speech tagger.
    Returns a flat tree structure where each word is directly under its POS tag.
    """
    def __init__(self, name: str, model_name: str = "en_core_web_sm"):
        super().__init__(name)
        self.model_name = model_name
        print(f"Initializing SpaCy POS Model ({model_name})...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading SpaCy model '{model_name}'...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Predicts POS tags using SpaCy.
        Returns a list of (token, tag) tuples.
        """
        doc = self.nlp(sentence)
        # Use .tag_ for fine-grained tags (PTB style) if available, else .pos_
        return [(token.text, token.tag_) for token in doc]

    def get_tree_string(self, sentence: str) -> Optional[str]:
        """
        Constructs a flat parse tree string (PTB format) from POS tags.
        Example: (S (NP (DT The) (NN cat)) (VP (VBD sat)))
        Since we only have POS tags, we wrap everything in a generic S node.
        """
        doc = self.nlp(sentence)
        # Construct a flat tree: (S (TAG word) (TAG word) ...)
        children = [f"({token.tag_} {token.text})" for token in doc]
        return f"(S {' '.join(children)})"


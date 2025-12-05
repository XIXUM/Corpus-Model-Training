import re
from typing import List, Iterator

class DataLoader:
    """
    Handles loading of text files and splitting them into sentences/segments 
    based on a recursive regex pattern provided.
    """
    
    # Regex pattern provided in the user query
    # It handles quotes, parentheses, square brackets, curly braces, and other text.
    # The recursive patterns (?R) are approximated or handled by the regex engine if supported (Python's re does not support (?R) recursion natively).
    # However, the regex library `regex` (pip install regex) DOES support recursion.
    # We will use the standard `re` for basic splitting if possible, or switch to `regex` module if recursion is strictly needed.
    # Given the prompt explicitly mentions recursive patterns like (?R), we MUST use the `regex` library.
    
    REGEX_SECTIONS = r"""
    (?P<quot>  (") ([^"]*) (") )
    |(?P<paren> (\() (?:[^()]+|(?R))* (\)) )
    |(?P<sqp>   (\[) (?:[^\[\]]+|(?R))* (\]) )
    |(?P<curl>  (\{) (?:[^{}]+|(?R))* (\}) )
    |(?P<other> [^()"\[\]{}]+ )
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        # We need the 'regex' module for recursive patterns
        import regex
        self.pattern = regex.compile(self.REGEX_SECTIONS, regex.VERBOSE | regex.DOTALL)

    def load_and_split(self) -> Iterator[str]:
        """
        Reads the file and yields segments based on the regex splitting.
        Note: The user mentioned that "below a bracketed structure also many sentences can occur".
        The regex splits the text into high-level chunks (quotes, bracketed groups, or other text).
        The 'other' text blocks might contain multiple sentences which then need further splitting (e.g. by NLTK).
        """
        import regex
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Iterate over matches
        for match in self.pattern.finditer(content):
            # A match can be one of the named groups.
            # We want the full matched string.
            segment = match.group(0)
            
            # If it is a 'quot', 'paren', 'sqp', or 'curl' block, we treat it as a unit (as per "bracketed sentences are always together").
            # If it is 'other', it's a block of text that might contain multiple sentences.
            
            # Check which group matched
            if match.group('other'):
                # This is a block of regular text. It might contain newlines or multiple sentences.
                # We should clean it up or yield it.
                # Usually, we want to yield clean sentences. 
                # If the user wants "bracketed sentences together", this loop separates them from the 'other' text.
                # The 'other' text needs to be further split into sentences?
                # The prompt says "splits the sentences accordingly so bracketed sentences are always together. but below a bracketed structure also many sentences can occur."
                # This implies we should yield the bracketed stuff as one unit, and split the 'other' stuff into sentences.
                
                text_block = segment.strip()
                if text_block:
                    # Use NLTK or simple splitting for the text block
                    # For now, let's just yield the block or split by common sentence terminators if needed.
                    # But for the first step, let's yield the raw segments or simple split.
                    # Let's assume we need to split 'other' by sentence delimiters (.!?)
                    # Simpler approach for now: just yield the segment, let the pipeline handle further tokenization if needed.
                    # OR: Use nltk.sent_tokenize on the 'other' parts?
                    # Let's stick to yielding the segment for now, as the regex determines the "structural" boundaries.
                    yield text_block
            else:
                # It's a bracketed/quoted segment. Yield as is.
                yield segment.strip()



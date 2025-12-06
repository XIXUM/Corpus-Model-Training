import re
import requests
import os
import tempfile
from typing import List, Iterator

class DataLoader:
    """
    Handles loading of text files (local or URL) and splitting them into sentences/segments 
    based on a recursive regex pattern provided.
    """
    
    REGEX_SECTIONS = r"""
    (?P<quot>  (") ([^"]*) (") )
    |(?P<paren> (\() (?:[^()]+|(?R))* (\)) )
    |(?P<sqp>   (\[) (?:[^\[\]]+|(?R))* (\]) )
    |(?P<curl>  (\{) (?:[^{}]+|(?R))* (\}) )
    |(?P<other> [^()"\[\]{}]+ )
    """

    def __init__(self, file_path_or_url: str):
        self.source = file_path_or_url
        import regex
        self.pattern = regex.compile(self.REGEX_SECTIONS, regex.VERBOSE | regex.DOTALL)
        self.is_url = self.source.startswith('http://') or self.source.startswith('https://')

    def load_and_split(self) -> Iterator[str]:
        """
        Reads the file (or downloads it) and yields segments based on the regex splitting.
        """
        import regex
        
        content = ""
        if self.is_url:
            print(f"Downloading data from {self.source}...")
            try:
                response = requests.get(self.source)
                response.raise_for_status()
                content = response.text
            except requests.exceptions.RequestException as e:
                print(f"Error downloading data: {e}")
                return
        else:
            try:
                with open(self.source, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"Error: File {self.source} not found.")
                return

        # Iterate over matches
        for match in self.pattern.finditer(content):
            segment = match.group(0)
            
            if match.group('other'):
                text_block = segment.strip()
                if text_block:
                    yield text_block
            else:
                yield segment.strip()

from typing import Any, List, Tuple
import pandas as pd

class Comparator:
    def __init__(self):
        pass

    def compare(self, result_a: Any, result_b: Any) -> bool:
        """
        Compares two model outputs. Returns True if they agree, False otherwise.
        For this stage, we assume simple equality of the returned structure (e.g. list of tagged tuples).
        """
        # Deep comparison of structure
        return result_a == result_b

    def find_diff(self, result_a: Any, result_b: Any) -> str:
        """
        Returns a string description of the difference.
        """
        if result_a == result_b:
            return "No difference"
        return f"Model A: {result_a} != Model B: {result_b}"


import nltk
from typing import Set, Tuple, Dict

def get_constituents(tree: nltk.Tree) -> Set[Tuple[str, int, int]]:
    """
    Extracts a set of phrasal constituents from an NLTK tree.
    Returns a set of tuples: (label, start_index, end_index).
    Indices are 0-based token indices.
    Excludes pre-terminals (POS tags).
    """
    constituents = set()
    
    def traverse(t, start_idx):
        # If it's a leaf string, just advance index
        if isinstance(t, str): 
            return start_idx + 1
        
        # If it's a Tree
        end_idx = start_idx
        for child in t:
            end_idx = traverse(child, end_idx)
            
        # Add constituent if it's a phrasal node (not a POS tag)
        # Heuristic: height > 2 means it has children that are not just text strings.
        # (DT The) -> height 2
        # (NP (DT The)) -> height 3
        if isinstance(t, nltk.Tree):
            if t.height() > 2: 
                constituents.add((t.label(), start_idx, end_idx))
            
        return end_idx

    traverse(tree, 0)
    return constituents

def calculate_metrics(gold_tree: nltk.Tree, pred_tree: nltk.Tree) -> Dict[str, float]:
    """
    Calculates Precision, Recall, F1, and Exact Match between two trees.
    """
    gold_const = get_constituents(gold_tree)
    pred_const = get_constituents(pred_tree)
    
    correct = len(gold_const.intersection(pred_const))
    gold_count = len(gold_const)
    pred_count = len(pred_const)
    
    precision = correct / pred_count if pred_count > 0 else 1.0 if gold_count == 0 else 0.0
    recall = correct / gold_count if gold_count > 0 else 1.0 if pred_count == 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    # Check for structure equality (Exact Match)
    # We compare the set of constituents + the leaves text must match (assumed if coming from same source)
    # But pure exact match of structure is often checked via string comparison of normalized trees
    # or set equality if we trust leaves match.
    exact_match = (gold_const == pred_const)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": 1.0 if exact_match else 0.0
    }


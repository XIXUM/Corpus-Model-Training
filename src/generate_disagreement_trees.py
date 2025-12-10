import json
import os
import glob
import sys
import argparse
from typing import List, Dict

# Add project root to path
sys.path.append(os.getcwd())

from src.models.benepar_wrapper import BeneparWrapper

def find_latest_log(log_dir: str = "disagreement_logs") -> str:
    files = glob.glob(os.path.join(log_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON logs found in {log_dir}")
    return max(files, key=os.path.getctime)

def load_sentences_from_log(log_path: str) -> List[str]:
    print(f"Loading sentences from: {log_path}")
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = [entry['sentence'] for entry in data]
    return sentences

def generate_trees(sentences: List[str], output_file: str):
    print("Initializing Benepar model...")
    # Assuming standard model; in a robust script we might parse model name from log
    model = BeneparWrapper(name="benepar", model_name="benepar_en3")
    
    print(f"Generating trees for {len(sentences)} sentences...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            # BeneparWrapper.get_tree_string returns the bracketed string
            tree_str = model.get_tree_string(sentence)
            if tree_str:
                f.write(tree_str + "\n")
            else:
                print(f"Warning: Could not generate tree for sentence: {sentence[:30]}...")
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(sentences)}")
                
    print(f"Successfully wrote {len(sentences)} trees to {output_file}")
    print("You can now manually correct these trees and use them for training.")

def main():
    parser = argparse.ArgumentParser(description="Generate PTB tree file from disagreement logs for manual correction.")
    parser.add_argument("--log-dir", type=str, default="disagreement_logs", help="Directory containing disagreement logs")
    parser.add_argument("--output", type=str, default="data/benepar_disagreements.ptb", help="Output file path for trees")
    
    args = parser.parse_args()
    
    try:
        latest_log = find_latest_log(args.log_dir)
        sentences = load_sentences_from_log(latest_log)
        generate_trees(sentences, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


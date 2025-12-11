import json
import os
import glob
import sys
import argparse
import nltk
from typing import List, Dict

# Add project root to path
sys.path.append(os.getcwd())

from src.models.benepar_wrapper import BeneparWrapper
from src.pipeline.data_loader import DataLoader

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

def load_sentences_from_source(source_file: str) -> List[str]:
    """
    Load and split sentences from a source file, applying the same logic
    as adversarial mode to handle quoted segments correctly.
    """
    print(f"Loading sentences from source file: {source_file}")
    loader = DataLoader(source_file)
    sentences = []
    
    for segment in loader.load_and_split():
        if not segment:
            continue
            
        is_special_block = segment[0] in ['"', '(', '[', '{']
        
        # For quoted segments, we need special handling to ensure all sentences are captured
        # even if they span multiple lines or contain multiple sentences
        if is_special_block and segment[0] == '"':
            # Normalize whitespace first (replace newlines with spaces for tokenization)
            quote_content = segment.strip()
            # If it starts and ends with quotes, extract the content
            if quote_content.startswith('"') and quote_content.endswith('"'):
                inner_content = quote_content[1:-1]  # Remove surrounding quotes
                # Replace newlines with spaces for proper sentence tokenization
                inner_content = inner_content.replace('\n', ' ').replace('\r', ' ')
                # Tokenize sentences within the quote
                sub_sentences = nltk.sent_tokenize(inner_content)
                # Re-add quotes to each sentence
                sub_sentences = [f'"{s.strip()}"' for s in sub_sentences if s.strip()]
            else:
                # Partial quote or malformed, use standard tokenization
                normalized = segment.replace('\n', ' ').replace('\r', ' ')
                sub_sentences = nltk.sent_tokenize(normalized)
        else:
            # Normalize newlines for non-quoted segments too
            normalized = segment.replace('\n', ' ').replace('\r', ' ')
            sub_sentences = nltk.sent_tokenize(normalized)
        
        for sentence in sub_sentences:
            sentence = sentence.replace('\n', ' ').strip()
            if sentence:
                sentences.append(sentence)
    
    print(f"Extracted {len(sentences)} sentences from source file")
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
    parser = argparse.ArgumentParser(description="Generate PTB tree file from disagreement logs or source file for manual correction.")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory containing disagreement logs (if not using --source)")
    parser.add_argument("--source", type=str, default=None, help="Source file to extract sentences from (uses same logic as adversarial mode)")
    parser.add_argument("--output", type=str, default="data/benepar_disagreements.ptb", help="Output file path for trees")
    
    args = parser.parse_args()
    
    try:
        # Download NLTK punkt if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
        
        if args.source:
            # Load from source file (handles quoted segments correctly)
            sentences = load_sentences_from_source(args.source)
        else:
            # Load from latest disagreement log
            log_dir = args.log_dir or "disagreement_logs"
            latest_log = find_latest_log(log_dir)
            sentences = load_sentences_from_log(latest_log)
        
        generate_trees(sentences, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


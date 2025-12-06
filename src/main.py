import argparse
import nltk
import regex  # type: ignore
import sys
import pandas as pd
import ast
from src.models.dummy_model import DummyModel
from src.models.benepar_wrapper import BeneparWrapper
from src.pipeline.comparator import Comparator
from src.pipeline.logger import DisagreementLogger
from src.pipeline.data_loader import DataLoader
from src.pipeline.tree_exporter import TreeExporter

def run_adversarial_mode(data_file: str, use_real_benepar: bool = False):
    """
    Runs the adversarial comparison between two models.
    """
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Initialize components
    if use_real_benepar:
        print("Initializing Benepar Wrapper (Real Model)...")
        model_a = BeneparWrapper("Model_A_Benepar")
    else:
        model_a = DummyModel("Model_A_Benepar", variation=True)
        
    model_b = DummyModel("Model_B_Adversarial", variation=False)
    
    comparator = Comparator()
    logger = DisagreementLogger(output_dir="disagreement_logs")
    tree_exporter = None
    
    if use_real_benepar:
        tree_exporter = TreeExporter(output_dir="disagreement_logs")
        print(f"Exporting trees to: {tree_exporter.get_file_path()}")

    print(f"Loading data from {data_file}...")
    
    try:
        loader = DataLoader(data_file)
        print("Starting adversarial evaluation...")
        
        for segment in loader.load_and_split():
            if not segment:
                continue
                
            is_special_block = segment[0] in ['"', '(', '[', '{']
            
            # Try to split everything into sentences, including quotes/brackets
            sub_sentences = nltk.sent_tokenize(segment)
                
            for sentence in sub_sentences:
                sentence = sentence.replace('\n', ' ').strip()
                if not sentence:
                    continue
                
                # Display and Export Tree for EVERY sentence if using real Benepar
                if use_real_benepar and isinstance(model_a, BeneparWrapper):
                    print(f"Processing: {sentence[:50]}...")
                    
                    # Display on console
                    model_a.display_tree(sentence)
                    
                    # Export to file
                    # Access the last computed tree string from the model if available
                    # Note: display_tree computes it again. 
                    # Let's optimize: call predict first to get POS, then check internal state or re-parse?
                    # BeneparWrapper.predict computes tags. 
                    # BeneparWrapper.display_tree computes tree.
                    # Ideally we get both.
                    
                    # Let's use display_tree logic to get the tree object string
                    # But display_tree prints directly.
                    # Let's add a get_tree_string method to BeneparWrapper
                    tree_str = model_a.get_tree_string(sentence)
                    if tree_exporter:
                        tree_exporter.log_tree(sentence, tree_str)

                res_a = model_a.predict(sentence)
                res_b = model_b.predict(sentence)
                
                # here the models are compared and only in case of a deviation the disagreement is logged
                if not comparator.compare(res_a, res_b):
                    diff = comparator.find_diff(res_a, res_b)
                    logger.log(sentence, model_a.name, res_a, model_b.name, res_b, diff)
                    
                    if use_real_benepar:
                         print(f" -> Disagreement found! (Logged)")

        logger.save()
        print("Adversarial evaluation complete.")
        
    except ImportError as e:
        print(f"Error: Missing dependency. {e}")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


def run_training_mode(csv_file: str):
    """
    Runs the training mode using the problematic sentences from the CSV.
    """
    print(f"Starting training mode using data from {csv_file}...")
    
    if not csv_file or not os.path.exists(csv_file):
        print("Error: Valid CSV file path is required for training mode.")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records for training.")
    
    epochs = 5
    for epoch in range(epochs):
        print(f"Training Epoch {epoch+1}/{epochs}...")
        for index, row in df.iterrows():
            pass
            
    print("Training complete.")


import os

def main():
    parser = argparse.ArgumentParser(description="Corpus Model Training & Adversarial Evaluation Tool")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Adversarial Mode
    parser_adv = subparsers.add_parser('adversarial', help='Run adversarial comparison')
    parser_adv.add_argument('--data', type=str, default="data/ASchoolEssay.txt", help='Path to input text file')
    parser_adv.add_argument('--real-benepar', action='store_true', help='Use real Benepar model instead of dummy')

    # Training Mode
    parser_train = subparsers.add_parser('train', help='Train on problematic sentences')
    parser_train.add_argument('--csv', type=str, required=True, help='Path to CSV file with disagreements')

    args = parser.parse_args()

    if args.mode == 'adversarial':
        run_adversarial_mode(args.data, args.real_benepar)
    elif args.mode == 'train':
        run_training_mode(args.csv)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

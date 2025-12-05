import argparse
import nltk
import regex  # type: ignore
import sys
import pandas as pd
import ast
from src.models.dummy_model import DummyModel
from src.pipeline.comparator import Comparator
from src.pipeline.logger import DisagreementLogger
from src.pipeline.data_loader import DataLoader

def run_adversarial_mode(data_file: str):
    """
    Runs the adversarial comparison between two models.
    """
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Initialize components
    model_a = DummyModel("Model_A_Benepar", variation=True)
    model_b = DummyModel("Model_B_Adversarial", variation=False)
    comparator = Comparator()
    logger = DisagreementLogger(output_dir="disagreement_logs")

    print(f"Loading data from {data_file}...")
    
    try:
        loader = DataLoader(data_file)
        print("Starting adversarial evaluation...")
        
        for segment in loader.load_and_split():
            if not segment:
                continue
                
            is_special_block = segment[0] in ['"', '(', '[', '{']
            
            sub_sentences = []
            if not is_special_block:
                sub_sentences = nltk.sent_tokenize(segment)
            else:
                sub_sentences = [segment]
                
            for sentence in sub_sentences:
                sentence = sentence.replace('\n', ' ').strip()
                if not sentence:
                    continue
                    
                res_a = model_a.predict(sentence)
                res_b = model_b.predict(sentence)
                
                if not comparator.compare(res_a, res_b):
                    diff = comparator.find_diff(res_a, res_b)
                    # Pass raw results to logger
                    logger.log(sentence, model_a.name, res_a, model_b.name, res_b, diff)

        logger.save()
        print("Adversarial evaluation complete.")
        
    except ImportError:
        print("Error: 'regex' module not found. Please install it via 'pip install regex'.")
    except FileNotFoundError:
        print(f"Error: File {data_file} not found.")


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
    
    # Mock training loop
    # In a real scenario, we would:
    # 1. Parse the 'result_b' (assuming B is the correct/adversarial one) or manual labels
    # 2. Convert to training tensors
    # 3. Train the model
    
    epochs = 5
    for epoch in range(epochs):
        print(f"Training Epoch {epoch+1}/{epochs}...")
        for index, row in df.iterrows():
            sentence = row['sentence']
            # Assuming Model B is the target/ground truth for now
            target_raw = row['result_b'] 
            
            # Parse the string representation back to list/object if needed
            # target_structure = ast.literal_eval(target_raw)
            
            # Simulate training step
            pass
            
    print("Training complete.")


import os

def main():
    parser = argparse.ArgumentParser(description="Corpus Model Training & Adversarial Evaluation Tool")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Adversarial Mode
    parser_adv = subparsers.add_parser('adversarial', help='Run adversarial comparison')
    parser_adv.add_argument('--data', type=str, default="data/ASchoolEssay.txt", help='Path to input text file')

    # Training Mode
    parser_train = subparsers.add_parser('train', help='Train on problematic sentences')
    parser_train.add_argument('--csv', type=str, required=True, help='Path to CSV file with disagreements')

    args = parser.parse_args()

    if args.mode == 'adversarial':
        run_adversarial_mode(args.data)
    elif args.mode == 'train':
        run_training_mode(args.csv)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

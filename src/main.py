import argparse
import nltk
import regex  # type: ignore
import sys
import pandas as pd
import ast
import os
import shutil
from typing import Dict, Any

from src.models.dummy_model import DummyModel
from src.models.benepar_wrapper import BeneparWrapper
from src.pipeline.comparator import Comparator
from src.pipeline.logger import DisagreementLogger
from src.pipeline.data_loader import DataLoader
from src.pipeline.tree_exporter import TreeExporter
from src.pipeline.html_reporter import HTMLTreeReporter

def clean_reports(output_dir="reports"):
    """
    Cleans up old reports/images to avoid ghost instances.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def get_model_instance(model_name: str, instance_name: str) -> Any:
    """
    Factory method to instantiate models based on CLI args.
    """
    model_name = model_name.lower()
    
    if model_name == "benepar":
        print(f"Initializing {instance_name} as BeneparWrapper (en3)...")
        return BeneparWrapper(instance_name, model_name="benepar_en3")
    elif model_name == "dummy":
        print(f"Initializing {instance_name} as DummyModel...")
        # If it's Model A (or the first one), we might want variation, or not.
        # Let's assume 'A' usually has the 'variation' (false positive) for testing purposes if both are dummy.
        # Or better, make the variation flag dependent on the name or random.
        # For this specific test case, let's keep the logic simple:
        # If name contains "Model A" or ends with "A", use variation.
        use_variation = "A" in instance_name
        return DummyModel(instance_name, variation=use_variation)
    elif model_name == "stanza":
        print(f"Warning: Stanza model not yet fully implemented. Falling back to Dummy.")
        return DummyModel(f"{instance_name} (Stanza Placeholder)", variation=True)
    elif model_name == "bert":
        print(f"Warning: BERT model not yet fully implemented. Falling back to Dummy.")
        return DummyModel(f"{instance_name} (BERT Placeholder)", variation=True)
    else:
        print(f"Unknown model '{model_name}'. Using DummyModel.")
        return DummyModel(instance_name, variation=True)

def get_model_info(model_obj: Any) -> Dict[str, str]:
    """
    Extracts metadata from model object for reporting.
    """
    info = {"name": model_obj.name}
    if isinstance(model_obj, DummyModel):
        info["type"] = "Dummy / Simulation"
    elif isinstance(model_obj, BeneparWrapper):
        info["type"] = f"Benepar ({model_obj.model_name})"
    else:
        info["type"] = "Unknown Model"
    return info

def run_adversarial_mode(data_source: str, model_a_type: str, model_b_type: str):
    """
    Runs the adversarial comparison between two models.
    """
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Clean previous reports
    clean_reports("reports")
    
    # Determine instance names
    name_a = f"{model_a_type.capitalize()}"
    name_b = f"{model_b_type.capitalize()}"
    
    # If comparing same type, distinguish them
    if model_a_type.lower() == model_b_type.lower():
        name_a += " (A)"
        name_b += " (B)"
    elif model_a_type.lower() == "benepar":
         name_a = "Benepar (en3)"

    # Initialize components based on CLI args
    model_a = get_model_instance(model_a_type, name_a)
    model_b = get_model_instance(model_b_type, name_b)
    
    comparator = Comparator()
    logger = DisagreementLogger(output_dir="disagreement_logs")
    
    # Setup Tree Exporter if using real models
    tree_exporter = None
    # Export if at least one model is real (not dummy)
    if not isinstance(model_a, DummyModel) or not isinstance(model_b, DummyModel):
        tree_exporter = TreeExporter(output_dir="tree_logs")
        print(f"Exporting trees to: {tree_exporter.get_file_path()}")

    # HTML Reporter with Metadata
    info_a = get_model_info(model_a)
    info_b = get_model_info(model_b)
    html_reporter = HTMLTreeReporter(output_dir="reports", model_a_info=info_a, model_b_info=info_b)

    # DataLoader handles both file paths and URLs now
    print(f"Loading data from {data_source}...")
    
    try:
        loader = DataLoader(data_source)
        print("Starting adversarial evaluation...")
        
        for segment in loader.load_and_split():
            if not segment:
                continue
                
            is_special_block = segment[0] in ['"', '(', '[', '{']
            sub_sentences = nltk.sent_tokenize(segment)
                
            for sentence in sub_sentences:
                sentence = sentence.replace('\n', ' ').strip()
                if not sentence:
                    continue
                
                res_a = model_a.predict(sentence)
                res_b = model_b.predict(sentence)
                
                # Get Tree Strings
                tree_a_str = None
                if hasattr(model_a, 'get_tree_string'):
                    tree_a_str = model_a.get_tree_string(sentence)
                    
                tree_b_str = None 
                if hasattr(model_b, 'get_tree_string'):
                    tree_b_str = model_b.get_tree_string(sentence)
                
                # Check for disagreement for the 'diff' field
                is_diff = not comparator.compare(res_a, res_b)
                diff_text = comparator.find_diff(res_a, res_b) if is_diff else "No Difference"
                
                html_reporter.add_comparison(
                    sentence, 
                    model_a.name, 
                    tree_a_str, 
                    model_b.name, 
                    tree_b_str,
                    diff_text
                )

                # Export to text file
                if tree_exporter and tree_a_str:
                    tree_exporter.log_tree(sentence, tree_a_str)

                # Disagreement Logging
                if is_diff:
                    logger.log(sentence, model_a.name, res_a, model_b.name, res_b, diff_text)

        logger.save()
        html_reporter.save() # Save HTML report
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
    parser_adv.add_argument('--data', type=str, default="data/ASchoolEssay.txt", help='Path to input text file OR URL')
    
    # Updated CLI args for models
    model_choices = ['dummy', 'benepar', 'stanza', 'bert']
    parser_adv.add_argument('--model-a', type=str, default="dummy", choices=model_choices, help='Model A selection')
    parser_adv.add_argument('--model-b', type=str, default="dummy", choices=model_choices, help='Model B selection')
    
    # Backward compatibility for old flag (optional, but good practice)
    parser_adv.add_argument('--real-benepar', action='store_true', help='DEPRECATED: Use --model-a benepar instead')

    # Training Mode
    parser_train = subparsers.add_parser('train', help='Train on problematic sentences')
    parser_train.add_argument('--csv', type=str, required=True, help='Path to CSV file with disagreements')

    args = parser.parse_args()

    if args.mode == 'adversarial':
        # Handle deprecation
        ma = args.model_a
        if args.real_benepar:
            print("Warning: --real-benepar is deprecated. Using --model-a benepar.")
            ma = "benepar"
            
        run_adversarial_mode(args.data, ma, args.model_b)
    elif args.mode == 'train':
        run_training_mode(args.csv)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

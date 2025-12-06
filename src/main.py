import argparse
import nltk
import regex  # type: ignore
import sys
import pandas as pd
import ast
import os
import shutil
from typing import Dict, Any, List, Optional

from src.models.dummy_model import DummyModel
from src.models.benepar_wrapper import BeneparWrapper
from src.models.stanza_wrapper import StanzaWrapper
from src.models.supar_wrapper import SuparWrapper
from src.pipeline.comparator import Comparator
from src.pipeline.logger import DisagreementLogger
from src.pipeline.data_loader import DataLoader
from src.pipeline.tree_exporter import TreeExporter
from src.pipeline.html_reporter import HTMLTreeReporter
from src.utils.metrics import calculate_metrics

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
    # Check if it's a file path first (for checkpoints)
    if os.path.exists(model_name):
        print(f"Initializing {instance_name} from local checkpoint: {model_name}")
        return BeneparWrapper(instance_name, model_name=model_name)

    model_name = model_name.lower()
    
    if model_name == "benepar":
        print(f"Initializing {instance_name} as BeneparWrapper (en3)...")
        return BeneparWrapper(instance_name, model_name="benepar_en3")
    elif model_name == "stanza":
        print(f"Initializing {instance_name} as StanzaWrapper...")
        return StanzaWrapper(instance_name)
    elif model_name == "supar":
        print(f"Initializing {instance_name} as SuPar (CRF-RoBERTa)...")
        return SuparWrapper(instance_name)
    elif model_name == "dummy":
        print(f"Initializing {instance_name} as DummyModel...")
        use_variation = "A" in instance_name
        return DummyModel(instance_name, variation=use_variation)
    elif model_name == "bert":
        print(f"Initializing {instance_name} as SuPar (BERT/RoBERTa-based)...")
        # Map 'bert' to SuPar wrapper with RoBERTa model (as standard BERT model key is deprecated/different)
        return SuparWrapper(instance_name, model_name="crf-con-roberta-en")
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
    elif isinstance(model_obj, StanzaWrapper):
        info["type"] = f"Stanza ({model_obj.model_name})"
    elif isinstance(model_obj, SuparWrapper):
        info["type"] = f"SuPar ({model_obj.model_name})"
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
    name_a = f"{model_a_type.capitalize()}" if not os.path.exists(model_a_type) else "Checkpoint Model A"
    name_b = f"{model_b_type.capitalize()}" if not os.path.exists(model_b_type) else "Checkpoint Model B"
    
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


def run_training_mode(train_data: str):
    """
    Runs the real training mode using the BeneparTrainer.
    """
    print(f"Starting REAL training mode using data from {train_data}...")
    
    if not train_data or not os.path.exists(train_data):
        print("Error: Valid training data file path is required.")
        return

    try:
        from src.training.trainer import BeneparTrainer
        trainer = BeneparTrainer()
        trainer.train(train_data)
    except ImportError as e:
        print(f"Error importing trainer: {e}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def run_cross_reference_mode(checkpoint_path: str, test_data: str):
    """
    Test whether the new trained model has fixed the false positives by comparing against reference data.
    """
    print(f"Starting Cross-Reference Verification...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Data: {test_data}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    if not os.path.exists(test_data):
        print(f"Error: Test data file not found at {test_data}")
        return

    # Initialize Trained Model
    try:
        # We use BeneparWrapper directly, passing the checkpoint path as model_name
        model = BeneparWrapper("Trained Model", model_name=checkpoint_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Load Test Data (Expects PTB Trees for verification)
    total = 0
    exact_matches = 0
    
    agg_precision = 0.0
    agg_recall = 0.0
    agg_f1 = 0.0

    print("\n--- Verification Results ---")
    try:
        with open(test_data, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    gold_tree = nltk.Tree.fromstring(line)
                    # Extract raw sentence from gold tree
                    sentence = " ".join(gold_tree.leaves())
                    
                    # Predict
                    pred_tree_str = model.get_tree_string(sentence)
                    if not pred_tree_str:
                        print(f"Warning: Failed to parse: {sentence[:30]}...")
                        continue
                        
                    pred_tree = nltk.Tree.fromstring(pred_tree_str)
                    
                    # Calculate Metrics
                    metrics = calculate_metrics(gold_tree, pred_tree)
                    
                    total += 1
                    exact_matches += metrics["exact_match"]
                    agg_precision += metrics["precision"]
                    agg_recall += metrics["recall"]
                    agg_f1 += metrics["f1"]
                        
                except ValueError:
                    print(f"Skipping invalid line in test data: {line[:30]}...")
                    continue
                    
    except Exception as e:
        print(f"Error reading test data: {e}")
        
    print("-" * 30)
    print(f"Total Sentences: {total}")
    
    if total > 0:
        avg_f1 = (agg_f1 / total) * 100
        avg_precision = (agg_precision / total) * 100
        avg_recall = (agg_recall / total) * 100
        exact_match_rate = (exact_matches / total) * 100
        
        print(f"Exact Match Accuracy: {exact_match_rate:.2f}%")
        print(f"Average F1 Score: {avg_f1:.2f}%")
        print(f"Average Precision: {avg_precision:.2f}%")
        print(f"Average Recall: {avg_recall:.2f}%")
        
        if avg_f1 < 99.0:
             print("\n Recommendation: Consider another training loop or reviewing the training data.")
        else:
             print("\n Success: Model performs with high accuracy!")
    else:
        print("No valid sentences processed.")


import os

def main():
    parser = argparse.ArgumentParser(description="Corpus Model Training & Adversarial Evaluation Tool")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Adversarial Mode
    parser_adv = subparsers.add_parser('adversarial', help='Run adversarial comparison')
    parser_adv.add_argument('--data', type=str, default="data/ASchoolEssay.txt", help='Path to input text file OR URL')
    
    # Updated choices for real models
    model_choices = ['dummy', 'benepar', 'stanza', 'supar', 'bert']
    parser_adv.add_argument('--model-a', type=str, default="dummy", choices=model_choices, help='Model A selection')
    parser_adv.add_argument('--model-b', type=str, default="dummy", choices=model_choices, help='Model B selection')
    parser_adv.add_argument('--real-benepar', action='store_true', help='DEPRECATED: Use --model-a benepar instead')

    # Training Mode
    parser_train = subparsers.add_parser('train', help='Train on problematic sentences')
    parser_train.add_argument('--train-data', type=str, required=True, help='Path to file with corrected parse trees (PTB format)')
    parser_train.add_argument('--csv', type=str, help='DEPRECATED: Use --train-data with tree file')

    # Cross-Reference Mode
    parser_cross = subparsers.add_parser('cross-reference', help='Verify trained model against reference data')
    parser_cross.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser_cross.add_argument('--test-data', type=str, required=True, help='Path to reference/gold standard trees (PTB format)')

    args = parser.parse_args()

    if args.mode == 'adversarial':
        ma = args.model_a
        if args.real_benepar:
            print("Warning: --real-benepar is deprecated. Using --model-a benepar.")
            ma = "benepar"
        run_adversarial_mode(args.data, ma, args.model_b)
        
    elif args.mode == 'train':
        data_path = args.train_data
        if args.csv:
            print("Warning: --csv is deprecated for training. Please provide a file with corrected trees using --train-data.")
            if not data_path:
                data_path = args.csv
        run_training_mode(data_path)
        
    elif args.mode == 'cross-reference':
        run_cross_reference_mode(args.checkpoint, args.test_data)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

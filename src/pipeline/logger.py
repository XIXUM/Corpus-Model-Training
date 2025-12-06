import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class DisagreementLogger:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file_csv = os.path.join(output_dir, f"disagreements_{timestamp}.csv")
        self.log_file_json = os.path.join(output_dir, f"disagreements_{timestamp}.json")
        self.data = []

    def log(self, sentence: str, model_a_name: str, result_a: Any, model_b_name: str, result_b: Any, diff_desc: str):
        """
        Logs a disagreement event.
        result_a and result_b are expected to be the raw output structure (e.g. list of tuples)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "sentence": sentence,
            "model_a": model_a_name,
            "result_a_str": str(result_a), # For CSV readability
            "result_a_raw": result_a,      # For JSON structure
            "model_b": model_b_name,
            "result_b_str": str(result_b),
            "result_b_raw": result_b,
            "difference": diff_desc
        }
        self.data.append(entry)

    def save(self):
        """
        Saves the accumulated logs to CSV and JSON.
        """
        if not self.data:
            print("No disagreements to save.")
            return

        # 1. Save CSV (simplified view)
        csv_data = []
        for item in self.data:
            csv_data.append({
                "timestamp": item["timestamp"],
                "sentence": item["sentence"],
                "model_a": item["model_a"],
                "result_a": item["result_a_str"],
                "model_b": item["model_b"],
                "result_b": item["result_b_str"],
                "difference": item["difference"]
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.log_file_csv, index=False)
        print(f"Disagreements saved to CSV: {self.log_file_csv}")

        # 2. Save JSON (structured view)
        # We need to format it as requested: "flat structure with dictionaries where they hold token:POSTag pair"
        # Currently our dummy model returns list of tuples: [('token', 'TAG'), ...]
        # We will convert this to dictionary { "token": "TAG", ... } 
        # WARNING: This flat dictionary structure loses duplicate tokens in a sentence!
        # e.g. "a cat saw a dog" -> {'a': 'DT', ...} -> The second 'a' overwrites the first? 
        # The user requested: "flat structure with dictionaries where they hold token:POSTag pair"
        # I should probably do a list of dictionaries or list of tokens with tags to be safe, 
        # but strictly following "token:POSTag pair" implies a dict. 
        # I will assume for now a list of objects is safer, but if they explicitly want {token: tag}, 
        # I'll try to conform but warn about duplicates?
        # Let's create a specific JSON structure for the training set.
        
        json_data = []
        for item in self.data:
            # Helper to convert list of tuples [('word', 'TAG'), ...] to dict {'word': 'TAG'}
            # If duplicates exist, this simple dict will collapse them.
            # Assuming for this stage we just want to see the mapping.
            
            # Process result_a
            tags_a = {}
            if isinstance(item["result_a_raw"], list):
                 for token, tag in item["result_a_raw"]:
                     tags_a[token] = tag
            
            # Process result_b
            tags_b = {}
            if isinstance(item["result_b_raw"], list):
                 for token, tag in item["result_b_raw"]:
                     tags_b[token] = tag

            json_entry = {
                "sentence": item["sentence"],
                "model_a": {
                    "name": item["model_a"],
                    "pos_tags": tags_a
                },
                "model_b": {
                    "name": item["model_b"],
                    "pos_tags": tags_b
                }
            }
            json_data.append(json_entry)

        with open(self.log_file_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
            
        print(f"Disagreements saved to JSON: {self.log_file_json}")

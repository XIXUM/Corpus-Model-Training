import pandas as pd
import os
from datetime import datetime

class DisagreementLogger:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.log_file = os.path.join(output_dir, f"disagreements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.data = []

    def log(self, sentence: str, model_a_name: str, result_a: str, model_b_name: str, result_b: str, diff_desc: str):
        """
        Logs a disagreement event.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "sentence": sentence,
            "model_a": model_a_name,
            "result_a": str(result_a),
            "model_b": model_b_name,
            "result_b": str(result_b),
            "difference": diff_desc
        }
        self.data.append(entry)

    def save(self):
        """
        Saves the accumulated logs to CSV.
        """
        if not self.data:
            print("No disagreements to save.")
            return

        df = pd.DataFrame(self.data)
        # Append if file exists? Or new file? user said "store such sentences... in a csv list"
        # We'll write a new file per session for now to avoid locking issues, or append.
        # Let's just write to the unique filename we created.
        df.to_csv(self.log_file, index=False)
        print(f"Disagreements saved to {self.log_file}")


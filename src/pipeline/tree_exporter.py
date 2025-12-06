import os
from datetime import datetime

class TreeExporter:
    """
    Exports constituency trees to a text file.
    """
    def __init__(self, output_dir: str = "tree_logs"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(output_dir, f"trees_{timestamp}.txt")
        
        # Clear/Create file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Constituency Tree Export - {timestamp}\n")
            f.write("="*50 + "\n\n")

    def log_tree(self, sentence: str, tree_str: str):
        """
        Appends a tree to the log file.
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"Sentence: {sentence}\n")
            if tree_str:
                # Pretty print via NLTK for file output?
                # Or just write the bracketed string?
                # Bracketed string is better for storage, pretty print for reading.
                # Let's try to pretty print if possible to file.
                try:
                    from nltk import Tree
                    import io
                    t = Tree.fromstring(tree_str)
                    
                    # Capture print output
                    f_str = io.StringIO()
                    t.pretty_print(stream=f_str)
                    f.write(f_str.getvalue())
                except:
                    f.write(f"{tree_str}\n")
            else:
                f.write("(No tree available)\n")
            
            f.write("-" * 40 + "\n")

    def get_file_path(self) -> str:
        return self.log_file

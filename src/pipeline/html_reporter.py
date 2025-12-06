import os
from datetime import datetime
import nltk
from nltk.draw.tree import TreeView
from nltk import Tree
import base64
import io

try:
    import svgling
except ImportError:
    svgling = None

class HTMLTreeReporter:
    """
    Generates an HTML report comparing two constituency trees side-by-side.
    Uses svgling (if available) or basic text representation.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Fixed filename for the latest run to avoid ghost files, as requested.
        self.report_file = os.path.join(output_dir, "latest_comparison_report.html")
        
        # Initialize report structure
        self.entries = []
        self.start_time = datetime.now()

    def add_comparison(self, sentence: str, model_a_name: str, tree_a_str: str, model_b_name: str, tree_b_str: str, diff_desc: str):
        """
        Adds a comparison entry to the report.
        """
        entry = {
            "sentence": sentence,
            "model_a": model_a_name,
            "tree_a": self._render_tree(tree_a_str),
            "model_b": model_b_name,
            "tree_b": self._render_tree(tree_b_str),
            "diff": diff_desc
        }
        self.entries.append(entry)

    def _render_tree(self, tree_str: str) -> str:
        """
        Renders a tree string to SVG or HTML format.
        """
        if not tree_str:
            return "<i>No tree available</i>"
            
        try:
            t = Tree.fromstring(tree_str)
            if svgling:
                # svgling.draw_tree returns a wrapper that can be converted to SVG
                # ._repr_svg_() returns the raw SVG string
                return svgling.draw_tree(t)._repr_svg_()
            else:
                # Fallback to text representation wrapped in pre
                return f"<pre>{t.pformat()}</pre>"
        except Exception as e:
            return f"<pre>Error rendering tree: {e}\n{tree_str}</pre>"

    def save(self):
        """
        Generates and saves the HTML file.
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Constituency Tree Comparison Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                .entry {{ border: 1px solid #ccc; margin-bottom: 20px; padding: 15px; border-radius: 5px; }}
                .sentence {{ font-weight: bold; font-size: 1.1em; margin-bottom: 10px; background-color: #f0f0f0; padding: 5px; }}
                .comparison {{ display: flex; gap: 20px; overflow-x: auto; }}
                .model-col {{ flex: 1; min-width: 300px; border: 1px solid #eee; padding: 10px; }}
                .model-name {{ font-weight: bold; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-bottom: 10px; }}
                .diff {{ color: #d9534f; margin-top: 10px; font-family: monospace; white-space: pre-wrap; }}
                svg {{ width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Comparison Report</h1>
            <p>Generated: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><b>Note:</b> Ensure 'svgling' is installed for graphical tree visualization.</p>
            
            <div id="entries">
        """
        
        for entry in self.entries:
            html_content += f"""
                <div class="entry">
                    <div class="sentence">{entry['sentence']}</div>
                    <div class="comparison">
                        <div class="model-col">
                            <div class="model-name">{entry['model_a']}</div>
                            {entry['tree_a']}
                        </div>
                        <div class="model-col">
                            <div class="model-name">{entry['model_b']}</div>
                            {entry['tree_b']}
                        </div>
                    </div>
                    <div class="diff">Difference: {entry['diff']}</div>
                </div>
            """
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"HTML Report saved to: {self.report_file}")

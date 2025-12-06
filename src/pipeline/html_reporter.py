import os
from datetime import datetime
import nltk
from nltk.draw.tree import TreeView
from nltk import Tree
import base64
import io
import ast

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
            
        self.report_file = os.path.join(output_dir, "latest_comparison_report.html")
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
            "diff": diff_desc,
            "diff_html": self._format_diff_table(diff_desc)
        }
        self.entries.append(entry)

    def _render_tree(self, tree_str: str) -> str:
        if not tree_str:
            return "<i>No tree available</i>"
        try:
            t = Tree.fromstring(tree_str)
            if svgling:
                return svgling.draw_tree(t)._repr_svg_()
            else:
                return f"<pre>{t.pformat()}</pre>"
        except Exception as e:
            return f"<pre>Error rendering tree: {e}\n{tree_str}</pre>"

    def _format_diff_table(self, diff_desc: str) -> str:
        """
        Parses the difference string and formats it as an HTML table comparison.
        Expects string format like: "Model A: [('Word', 'TAG'), ...] != Model B: [('Word', 'TAG'), ...]"
        """
        if diff_desc == "No Difference":
             return "<span class='no-diff'>No Difference</span>"
             
        try:
            # Simple parsing strategy based on known format
            if "Model A:" in diff_desc and "Model B:" in diff_desc:
                parts = diff_desc.split("!= Model B:")
                part_a = parts[0].replace("Model A:", "").strip()
                part_b = parts[1].strip()
                
                # Try to parse as list of tuples
                list_a = ast.literal_eval(part_a)
                list_b = ast.literal_eval(part_b)
                
                html = "<table class='diff-table'><thead><tr><th>Token</th><th>Model A Tag</th><th>Model B Tag</th></tr></thead><tbody>"
                
                # Iterate max length
                max_len = max(len(list_a), len(list_b))
                for i in range(max_len):
                    token_a, tag_a = list_a[i] if i < len(list_a) else ("-", "-")
                    token_b, tag_b = list_b[i] if i < len(list_b) else ("-", "-")
                    
                    # Highlight row if tags differ
                    row_class = "diff-row" if tag_a != tag_b else ""
                    
                    html += f"<tr class='{row_class}'><td>{token_a}</td><td>{tag_a}</td><td>{tag_b}</td></tr>"
                    
                html += "</tbody></table>"
                return html
                
        except Exception as e:
            # Fallback if parsing fails
            return f"<pre>{diff_desc}</pre>"
            
        return f"<pre>{diff_desc}</pre>"

    def save(self):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Constituency Tree Comparison Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f9f9f9; }}
                .entry {{ background: white; border: 1px solid #e0e0e0; margin-bottom: 30px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .sentence {{ font-weight: 600; font-size: 1.2em; margin-bottom: 15px; padding: 10px; background-color: #f0f7ff; border-left: 4px solid #0066cc; border-radius: 4px; }}
                
                .comparison-container {{ display: flex; flex-direction: column; gap: 20px; }}
                
                /* Tag Comparison Table */
                .diff-section {{ margin-bottom: 20px; }}
                .diff-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95em; }}
                .diff-table th {{ background-color: #f5f5f5; padding: 8px; text-align: left; border-bottom: 2px solid #ddd; }}
                .diff-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                .diff-row {{ background-color: #fff0f0; }}
                .diff-row td {{ color: #d9534f; font-weight: bold; }}
                .no-diff {{ color: #28a745; font-weight: bold; }}

                /* Tree Visualization */
                .trees-row {{ display: flex; gap: 20px; border-top: 1px solid #eee; padding-top: 20px; }}
                .model-col {{ flex: 1; min-width: 300px; background: #fff; padding: 10px; border: 1px solid #eee; border-radius: 4px; }}
                .model-name {{ font-weight: bold; color: #555; text-align: center; padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid #eee; text-transform: uppercase; font-size: 0.9em; letter-spacing: 1px; }}
                
                svg {{ width: 100%; height: auto; min-height: 200px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f8f8f8; padding: 10px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>Constituency Tree Comparison Report</h1>
            <p>Generated: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div id="entries">
        """
        
        for entry in self.entries:
            html_content += f"""
                <div class="entry">
                    <div class="sentence">{entry['sentence']}</div>
                    
                    <div class="comparison-container">
                        <!-- Tag Differences Table -->
                        <div class="diff-section">
                            <h3>POS Tag Comparison</h3>
                            {entry['diff_html']}
                        </div>
                        
                        <!-- Tree Visualizations -->
                        <div class="trees-row">
                            <div class="model-col">
                                <div class="model-name">{entry['model_a']} Tree</div>
                                {entry['tree_a']}
                            </div>
                            <div class="model-col">
                                <div class="model-name">{entry['model_b']} Tree</div>
                                {entry['tree_b']}
                            </div>
                        </div>
                    </div>
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

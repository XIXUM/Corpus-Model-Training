# Architecture Documentation

This document outlines the architecture, design decisions, and implemented features of the Corpus Model Training tool.

## Requirements & Implementation Status

The following core requirements have been addressed:

*   **Adversarial Training** (Implemented):
    *   **Mechanism**: Disagreements between models (e.g., Benepar vs. Dummy) are detected in `adversarial` mode and logged to CSV/JSON.
    *   **Training**: A dedicated `train` mode uses `BeneparTrainer` to retrain the model on corrected data derived from these disagreements.
    *   **Workflow**:
        1.  Run `adversarial` to find issues.
        2.  Correct the problematic sentences (create a Gold Standard file).
        3.  Run `train` with the corrected file.

*   **Tree Distance** (Implemented):
    *   **Metrics**: Standard PARSEVAL metrics (Precision, Recall, F1) and Exact Match accuracy are implemented in `src/utils/metrics.py`.
    *   **Usage**: These metrics are used in the `cross-reference` mode to quantitatively evaluate the retrained model against a reference dataset.

## System Architecture

### 1. Abstraction Layer
The system uses an abstraction layer for different parsing models to ensure modularity.

*   **`BaseModel`**: Abstract base class defining the interface (`predict`, `get_tree_string`).
*   **`DummyModel`**: A reference implementation for testing pipeline flow without heavy model dependencies.
*   **`BeneparWrapper`**: A wrapper around the real Benepar model (via spaCy). 
    *   Includes a `safe_benepar_parser` wrapper to handle `StopIteration` and other parsing errors robustly.
    *   Manages the spaCy pipeline, ensuring components like `sentencizer` are correctly configured.
*   **`SuparWrapper`**: A wrapper for SuPar's CRF Constituency Parser (supports BERT-based models).
    *   Includes patches for PyTorch 2.6+ serialization and tokenizer compatibility.
*   **`SpacyPOSModel`**: A simple POS tagging model using SpaCy.
    *   Provides explicit POS tags without deep constituency structure (generates a flat tree for visualization).
    *   Useful for training strictly on POS tags.

### 2. Pipeline Components
*   **Data Loader**: Handles reading raw text from local files or URLs. Uses Regex to split text while preserving nested structures (quotes, brackets).
*   **Comparator**: Compares outputs from two models to detect disagreements in POS tags or tree structure.
*   **Logger**: Logs disagreements to `disagreement_logs/` in both CSV and JSON formats.
*   **HTML Reporter**: Generates `reports/latest_comparison_report.html` with side-by-side SVG tree visualizations.
*   **Tree Exporter**: Exports raw text representations of trees to `tree_logs/`.

### 3. Main Execution (`src/main.py`)
The entry point handles CLI arguments to select the operation mode.

*   **Adversarial Mode**:
    *   Compares two models (e.g., Benepar vs. Dummy).
    *   Logs disagreements and generates visual reports.
    *   Exports trees.
*   **Training Mode**:
    *   Loads a file of corrected parse trees (PTB format).
    *   Runs the `BeneparTrainer` to fine-tune the model.
*   **Cross-Reference Mode**:
    *   Verifies a trained model checkpoint against a Gold Standard dataset.
    *   Reports accuracy metrics (F1, Exact Match).

### 4. Training Subsystem
*   **`BeneparTrainer`**:
    *   Located in `src/training/trainer.py`.
    *   Loads the underlying PyTorch `ChartParser`.
    *   Performs fine-tuning using standard gradient descent.
    *   Saves checkpoints to `checkpoints/`.
*   **Configuration**: Managed via `config/training_config.yaml` (Hardware: CUDA/MPS/CPU, Hyperparameters).

## Data Flow

### Adversarial Flow
1.  **Input**: Text file or URL (e.g., `data/ASchoolEssay.txt`).
2.  **Splitting**: Regex + NLTK splitting into sentences.
3.  **Processing**:
    *   Model A (Benepar) -> Prediction A & Tree A
    *   Model B (Adversarial) -> Prediction B & Tree B
4.  **Comparison**: Models' outputs are compared.
5.  **Reporting**:
    *   If Disagreement -> Log to `disagreement_logs/`.
    *   Always -> Add to HTML Report (`reports/`) and Text Export (`tree_logs/`).

### Training Flow
1.  **Input**: File with corrected trees (PTB format).
2.  **Loading**: `BeneparTrainer` parses trees into `InputExample` objects.
3.  **Training**:
    *   Model loaded (from `benepar_en3` or checkpoint).
    *   Batches created and passed to model.
    *   Loss computed -> Backprop -> Optimizer Step.
4.  **Output**: Checkpoint saved to `checkpoints/`.

### Cross-Reference Flow
1.  **Input**: Trained Checkpoint + Gold Standard Tree File.
2.  **Processing**:
    *   Load model from checkpoint.
    *   For each gold tree: extract text -> predict -> compare structure.
3.  **Output**: Evaluation metrics (Precision, Recall, F1, Exact Match).

## Future Work
*   Integrate Stanza and BERT models fully (currently placeholders).
*   Automate the "Correction" step (e.g., UI for fixing trees).

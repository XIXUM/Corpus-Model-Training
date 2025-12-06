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

## System Components

### 1. Pipeline
*   **Data Loader**: Handles reading raw text from local files or URLs.
*   **Comparator**: Compares outputs from two models.
*   **Logger**: Logs disagreements to `disagreement_logs/`.
*   **HTML Reporter**: Generates `reports/latest_comparison_report.html` with SVG tree visualizations.

### 2. Models
*   **BeneparWrapper**: Wraps the Benepar parser (via spaCy). Supports loading pretrained models or local checkpoints.
*   **DummyModel**: Simulates a parser for testing the pipeline.

### 3. Training
*   **BeneparTrainer**:
    *   Located in `src/training/trainer.py`.
    *   Loads the underlying PyTorch `ChartParser`.
    *   Performs fine-tuning using standard gradient descent.
    *   Saves checkpoints to `checkpoints/`.
*   **Configuration**: Managed via `config/training_config.yaml` (Hardware: CUDA/MPS/CPU, Hyperparameters).

### 4. Verification (Cross-Reference)
*   **Mode**: `cross-reference`.
*   **Function**: Loads a specific checkpoint and compares its predictions against a "Gold Standard" file of bracketed trees.
*   **Output**: Reports F1 Score, Precision, Recall, and Accuracy.

## Future Work
*   Integrate Stanza and BERT models fully (currently placeholders).
*   Automate the "Correction" step (e.g., UI for fixing trees).

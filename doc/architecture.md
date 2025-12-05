# System Architecture

## Overview

The goal of this project is to identify and correct false positives in constituency parsing models (specifically identifying where models like Benepar fail on POS tagging or tree structure) by using an adversarial/comparative approach.

The system runs in two primary modes:
1.  **Adversarial Mode**: Compares two models to identify and log disagreements.
2.  **Training Mode**: Uses the logged disagreements to retrain a model (simulated).

## Components

### 1. Pipeline (`src/pipeline/`)

The core logic for processing sentences.

*   **`Comparator`**: Compares the outputs of two models. Currently checks for structural equality.
*   **`DisagreementLogger`**: Handles the storage of flagged sentences.
    *   **CSV Output**: Saves the sentence and raw differences for easy reading and training.
    *   **JSON Output**: Saves a structured, flat dictionary of token-to-POS tags for each model, facilitating detailed analysis.
*   **`DataLoader`**: Handles loading text from files and splitting it into processing units.
    *   Uses a recursive Regex pattern to handle complex text structures like nested parentheses, quotes, etc.
    *   Ensures that bracketed/quoted sentences are kept together as single units.
    *   Normal text is split into sentences using NLTK.

### 2. Models (`src/models/`)

Abstraction layer for different parsing models.

*   **`BaseModel`**: Abstract base class defining the `predict(sentence)` interface.
*   **`DummyModel`**: A reference implementation used for testing the pipeline. It simulates a model with known "false positive" behaviors (e.g., misclassifying "today").

### 3. Main Execution (`src/main.py`)

The entry point that handles CLI arguments to select the mode.

*   **Adversarial Mode**:
    1.  Initializes models.
    2.  Loads data using `DataLoader`.
    3.  Logs disagreements to CSV and JSON.
*   **Training Mode**:
    1.  Loads a disagreement CSV.
    2.  Iterates through records (simulating a training loop).

## Data Flow

### Adversarial Flow
1.  **Input**: Text file (e.g., `data/ASchoolEssay.txt`).
2.  **Splitting**: Regex + NLTK splitting.
3.  **Comparison**: Model A vs Model B.
4.  **Output**: 
    *   `disagreements_TIMESTAMP.csv`
    *   `disagreements_TIMESTAMP.json` (contains flat `{token: tag}` structures)

### Training Flow
1.  **Input**: CSV file from the adversarial step.
2.  **Processing**: Extract sentence and target label (Model B output).
3.  **Training**: Update model weights (Simulated).

## Future Extensions

*   **Adversarial Training**: Use the collected disagreements to retrain the failing model.
*   **Tree Distance**: Implement more sophisticated tree comparison metrics (e.g., Evalb).
*   **Model Wrappers**: Implement actual wrappers for Benepar, Spacy, or CoreNLP.

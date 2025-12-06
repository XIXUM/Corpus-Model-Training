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
    *   **JSON Output**: Saves a structured, flat dictionary of token-to-POS tags for each model.
*   **`DataLoader`**: Handles loading text from files and splitting it into processing units using Regex.
*   **`POSDataLoader`**: Helper to load reference POS tags from CSV files.
*   **`TreeExporter`**: Exports the constituency trees of all processed sentences to a clean text file.
*   **`HTMLTreeReporter`**: **(New)** Generates a graphical HTML report comparing trees side-by-side.
    *   Uses `svgling` for high-quality SVG tree rendering.
    *   Cleaned up on each run to avoid "ghost" files.

### 2. Models (`src/models/`)

Abstraction layer for different parsing models.

*   **`BaseModel`**: Abstract base class.
*   **`DummyModel`**: A reference implementation for testing.
*   **`BeneparWrapper`**: A wrapper around the real Benepar model (via spaCy). 
    *   Includes a `safe_benepar_parser` wrapper to handle `StopIteration` errors robustly.
    *   Supports displaying constituency trees using NLTK.

### 3. Main Execution (`src/main.py`)

The entry point that handles CLI arguments to select the mode.

*   **Adversarial Mode**:
    *   Can use `DummyModel` (fast, no downloads) or `BeneparWrapper` (requires model download).
    *   Logs disagreements to CSV/JSON.
    *   Exports all trees to text file.
    *   Generates `reports/latest_comparison_report.html` with side-by-side tree visualizations.
*   **Training Mode**:
    *   Loads a disagreement CSV.
    *   Runs a mock training loop.

## Data Flow

### Adversarial Flow
1.  **Input**: Text file (e.g., `data/ASchoolEssay.txt`).
2.  **Splitting**: Regex + NLTK splitting.
3.  **Processing**:
    *   Model A (Benepar) -> Prediction A & Tree A
    *   Model B (Adversarial) -> Prediction B & Tree B
4.  **Reporting**:
    *   **Text Export**: `tree_logs/`
    *   **HTML Report**: `reports/` (Side-by-side SVG)
    *   **Comparison**: `Comparator`
    *   **Logging**: `disagreement_logs/` (CSV/JSON)

## Future Extensions

*   **Adversarial Training**: Use the collected disagreements to retrain the failing model.
*   **Tree Distance**: Implement more sophisticated tree comparison metrics (e.g., Evalb).

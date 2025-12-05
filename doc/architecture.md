# System Architecture

## Overview

The goal of this project is to identify and correct false positives in constituency parsing models (specifically identifying where models like Benepar fail on POS tagging or tree structure) by using an adversarial/comparative approach.

The system runs two models in parallel on the same corpus. Deviations in their output (POS tags or Constituency Trees) are flagged and logged to a CSV file for further analysis or retraining.

## Components

### 1. Pipeline (`src/pipeline/`)

The core logic for processing sentences.

*   **`Comparator`**: Compares the outputs of two models. Currently checks for structural equality.
*   **`DisagreementLogger`**: Handles the storage of flagged sentences. It saves:
    *   The input sentence.
    *   Output of Model A.
    *   Output of Model B.
    *   Description of the difference.
    *   Timestamp.
*   **`DataLoader`**: Handles loading text from files and splitting it into processing units.
    *   Uses a recursive Regex pattern to handle complex text structures like nested parentheses, quotes, etc.
    *   Ensures that bracketed/quoted sentences are kept together as single units.
    *   Normal text is split into sentences using NLTK.

### 2. Models (`src/models/`)

Abstraction layer for different parsing models.

*   **`BaseModel`**: Abstract base class defining the `predict(sentence)` interface.
*   **`DummyModel`**: A reference implementation used for testing the pipeline. It simulates a model with known "false positive" behaviors (e.g., misclassifying "today").

### 3. Main Execution (`src/main.py`)

The entry point that:
1.  Initializes the models.
2.  Loads data using `DataLoader`.
3.  Iterates through the segments/sentences.
4.  Invokes the comparator.
5.  Logs disagreements.

## Data Flow

1.  **Input**: Text file (e.g., `data/ASchoolEssay.txt`).
2.  **Splitting**:
    *   Regex splits text into segments (Quotes, Brackets, Normal Text).
    *   Normal Text segments are further tokenized into sentences.
3.  **Processing**:
    *   Model A -> Prediction A
    *   Model B -> Prediction B
4.  **Comparison**: `Comparator.compare(Prediction A, Prediction B)`
5.  **Decision**:
    *   **Match**: No action.
    *   **Mismatch**: Log details to CSV.
6.  **Output**: CSV file containing disagreements.

## Future Extensions

*   **Adversarial Training**: Use the collected disagreements to retrain the failing model.
*   **Tree Distance**: Implement more sophisticated tree comparison metrics (e.g., Evalb).
*   **Model Wrappers**: Implement actual wrappers for Benepar, Spacy, or CoreNLP.

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
*   **`DataLoader`** (To be implemented): Will handle loading large corpora from files.

### 2. Models (`src/models/`)

Abstraction layer for different parsing models.

*   **`BaseModel`**: Abstract base class defining the `predict(sentence)` interface.
*   **`DummyModel`**: A reference implementation used for testing the pipeline. It simulates a model with known "false positive" behaviors (e.g., misclassifying "today").

### 3. Main Execution (`src/main.py`)

The entry point that:
1.  Initializes the models.
2.  Iterates through the corpus.
3.  Invokes the comparator.
4.  Logs disagreements.

## Data Flow

1.  **Input**: Sentence from Corpus.
2.  **Processing**:
    *   Model A -> Prediction A
    *   Model B -> Prediction B
3.  **Comparison**: `Comparator.compare(Prediction A, Prediction B)`
4.  **Decision**:
    *   **Match**: No action.
    *   **Mismatch**: Log details to CSV.
5.  **Output**: CSV file containing disagreements.

## Future Extensions

*   **Adversarial Training**: Use the collected disagreements to retrain the failing model.
*   **Tree Distance**: Implement more sophisticated tree comparison metrics (e.g., Evalb).
*   **Model Wrappers**: Implement actual wrappers for Benepar, Spacy, or CoreNLP.


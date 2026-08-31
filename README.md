# Corpus Model Training & Adversarial Evaluation Tool

## Overview
This tool allows for the adversarial evaluation of constituency parsing models (e.g., Benepar, SuPar) and provides a framework for retraining models on problematic sentences. It identifies disagreements between two models, logs them, and generates visual comparison reports.

## Features
- **Adversarial Evaluation:** Compares outputs of two models.
- **Model Selection:**
  - `benepar`: Berkeley Neural Parser (Chart-based, Transformer).
  - `stanza`: Stanford NLP (Constituency).
  - `supar`: SuPar CRF Constituency Parser (BERT-based).
  - `bert`: Alias for SuPar with `crf-con-bert-en` model.
  - `spacy_pos`: Simple POS tagger using SpaCy (flat tree structure).
  - `dummy`: Testing model.
- **Data Loading:** Supports loading training data from local text files or URLs.
- **Reporting:** HTML reports with SVG trees, CSV/JSON logs.
- **Training Mode:** Fine-tunes the Benepar model using corrected parse trees.
- **Cross-Reference Mode:** Verifies a trained model against a reference/gold-standard dataset.
- **Hardware Configuration:** Switch between `cuda`, `mps`, and `cpu`.

## Setup

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Configure hardware and training parameters in `config/training_config.yaml`:

```yaml
hardware:
  device: "mps"  # Options: cuda, mps, cpu

training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
```

## Usage

### 1. Adversarial Mode (Comparison)

Compare two models on a dataset. You can provide a local file path or a direct URL to a text file.

**Real Adversarial Example (Benepar vs. BERT-based SuPar) with Local File:**
```bash
python -m src.main adversarial --model-a benepar --model-b bert --data data/ASchoolEssay.txt
```

**Adversarial Example with URL Source:**
```bash
python -m src.main adversarial --model-a benepar --model-b dummy --data https://www.gutenberg.org/files/11/11-0.txt
```

### 2. Training Mode

Retrain the Benepar model on corrected sentences. Requires a file with bracketed parse trees (PTB format).

```bash
python -m src.main train --train-data data/corrected_trees.txt
```

### 3. Generating Training Data (Tree Extraction)

Extract parse trees from the latest disagreement log to create a starting point for manual correction.

```bash
python -m src.generate_disagreement_trees --output data/benepar_disagreements.ptb
```

Then, manually edit `data/benepar_disagreements.ptb` to correct the trees before running the training mode.

**Reruns preserve your corrections.** If the output file already exists, the
generator keeps every tree that is already there (matched to its sentence, even
if a correction edited a token) and only appends freshly generated trees for
sentences that are not yet represented. A timestamped `.bak` backup of the
previous file is written on every run. Pass `--force` to regenerate all trees
from scratch and discard manual corrections (the `.bak` backup is still made):

```bash
python -m src.generate_disagreement_trees --force
```

### 4. Cross-Reference Mode (Verification)

After training, verify if the false positives are fixed by comparing the new model's output against a reference dataset (Gold Standard).

```bash
python -m src.main cross-reference --checkpoint checkpoints/benepar_epoch_5.pt --test-data data/gold_standard.txt
```

This will output accuracy statistics and indicate if another training loop is recommended.

## Project Structure

- `src/`: Source code.
  - `models/`: Model wrappers (Benepar, Stanza, SuPar, Dummy).
  - `pipeline/`: Core logic (`DataLoader`, `Comparator`, `Logger`, `HTMLReporter`).
  - `training/`: Training logic.
  - `utils/`: Utilities.
  - `main.py`: Entry point.
- `config/`: Configuration.
- `data/`: Input data files.
- `reports/`: Generated HTML reports.
- `checkpoints/`: Saved model checkpoints.

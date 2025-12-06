# Corpus Model Training & Adversarial Evaluation Tool

## Overview
This tool allows for the adversarial evaluation of constituency parsing models (e.g., Benepar, Dummy models) and provides a framework for retraining models on problematic sentences. It identifies disagreements between two models, logs them, and generates visual comparison reports.

## Features
- **Adversarial Evaluation:** Compares outputs of two models (e.g., Benepar vs. Dummy).
- **Model Selection:** Choose from different models (`benepar`, `dummy`, `stanza`*, `bert`*) via CLI. (*placeholders)
- **Data Loading:** Supports loading training data from local text files or URLs.
- **Reporting:**
  - **HTML Report:** Side-by-side visual comparison of constituency trees (SVG) and POS tags.
  - **CSV/JSON Logs:** Detailed logs of disagreements.
  - **Tree Export:** Raw text export of parsed trees.
- **Training Mode:** (Stub) Framework for retraining on identified problematic sentences.

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
   *Note: `benepar` requires model download (handled automatically by the wrapper).*

## Usage

The tool operates in two main modes: `adversarial` and `train`.

### Adversarial Mode (Comparison)

Compare two models on a dataset.

**Basic Usage (Dummy Models):**
```bash
python -m src.main adversarial --data data/ASchoolEssay.txt
```

**Using Benepar (Real Model):**
```bash
python -m src.main adversarial --model-a benepar --model-b dummy --data data/ASchoolEssay.txt
```

**Using URL Data:**
```bash
python -m src.main adversarial --model-a dummy --model-b dummy --data https://www.gutenberg.org/cache/epub/11/pg11.txt
```

**Arguments:**
- `--data`: Path to local text file OR URL (default: `data/ASchoolEssay.txt`).
- `--model-a`: Select first model (`dummy`, `benepar`, `stanza`, `bert`).
- `--model-b`: Select second model (`dummy`, `benepar`, `stanza`, `bert`).

### Training Mode

Retrain on problematic sentences (requires a CSV from adversarial run).

```bash
python -m src.main train --csv disagreement_logs/disagreements_YYYYMMDD_HHMMSS.csv
```

## Project Structure

- `src/`: Source code.
  - `models/`: Model wrappers (`BaseModel`, `DummyModel`, `BeneparWrapper`).
  - `pipeline/`: Core logic (`DataLoader`, `Comparator`, `Logger`, `HTMLReporter`).
  - `main.py`: Entry point.
- `data/`: Input data files.
- `reports/`: Generated HTML reports.
- `disagreement_logs/`: CSV/JSON logs of disagreements.
- `tree_logs/`: Text export of trees.

## Output

- **HTML Report:** `reports/latest_comparison_report.html` - Open in browser to view tree visualizations.
- **Disagreement Logs:** Saved in `disagreement_logs/`.
- **Tree Logs:** Saved in `tree_logs/`.

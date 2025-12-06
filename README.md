# Corpus Model Training & Adversarial Evaluation

This project provides a framework for training and evaluating constituency parsing models. It specifically targets the identification of false positives (e.g., incorrect POS tagging in Benepar) by comparing outputs against an adversarial or reference model.

## Project Structure

```
.
├── src/
│   ├── main.py             # Entry point (CLI)
│   ├── models/             # Model wrappers (Dummy, Benepar)
│   └── pipeline/           # Core logic (Loader, Comparator, Logger)
├── doc/
│   └── architecture.md     # System architecture documentation
├── data/
│   └── ASchoolEssay.txt    # Sample data
├── disagreement_logs/      # Output CSV/JSON logs
├── requirements.txt        # Python dependencies
└── train.py                # (Legacy) Simple NN training script
```

## Setup

1.  Create a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Benepar requires model downloading. The wrapper attempts to download 'benepar_en3' automatically, but this requires internet access.*

## Usage

The tool now supports two modes: `adversarial` and `train`.

### 1. Adversarial Evaluation Mode

Run the comparison pipeline to detect deviations between models.

```bash
source .venv/bin/activate
python -m src.main adversarial --data data/ASchoolEssay.txt
```

To use the **real Benepar model** (requires `benepar_en3` download):
```bash
python -m src.main adversarial --data data/ASchoolEssay.txt --real-benepar
```
*This will also display the constituency trees for mismatched sentences.*

### 2. Training Mode

Run the training loop using the detected disagreements.

```bash
python -m src.main train --csv disagreement_logs/disagreements_YOUR_TIMESTAMP.csv
```

## Documentation

See `doc/architecture.md` for details on the system design, data flow, and JSON structure.

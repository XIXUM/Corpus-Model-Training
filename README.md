# Corpus Model Training & Adversarial Evaluation

This project provides a framework for training and evaluating constituency parsing models. It specifically targets the identification of false positives (e.g., incorrect POS tagging in Benepar) by comparing outputs against an adversarial or reference model.

## Project Structure

```
.
├── src/
│   ├── main.py             # Entry point for evaluation pipeline
│   ├── models/             # Model wrappers and base classes
│   │   ├── base_model.py   # Abstract base class
│   │   └── dummy_model.py  # Test model implementation
│   └── pipeline/           # Core logic
│       ├── comparator.py   # Logic to compare model outputs
│       └── logger.py       # CSV logging for disagreements
├── doc/
│   └── architecture.md     # System architecture documentation
├── disagreement_logs/      # Output folder for CSV logs
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

## Usage

### Adversarial Evaluation

To run the comparison pipeline which detects deviations between two models (currently simulated):

```bash
source .venv/bin/activate
python -m src.main
```

This will:
1.  Run two models on a sample corpus.
2.  Compare their outputs.
3.  Save any sentences where the models disagree to `disagreement_logs/`.

### Neural Network Training (Basic)

To run the basic neural network training script:

```bash
python train.py --epochs 50
```

## Documentation

See `doc/architecture.md` for details on the system design and components.

# Neural Network Training Tool

This project contains tools for training neural networks, specifically focused on NLP tasks using NLTK.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main training script is `train.py`. You can run it with the following arguments:

```bash
python train.py --epochs 100 --lr 0.01 --hidden 20
```

### Arguments:

- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.01)
- `--hidden`: Hidden layer size (default: 20)

### Example:

```bash
source .venv/bin/activate
python train.py --epochs 50
```

This will train a simple neural network on dummy data and demonstrate NLTK tokenization.

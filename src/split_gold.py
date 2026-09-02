"""
Split a PTB gold-tree file into a train and a held-out test split.

An honest cross-reference score needs a test set the model was NOT trained on.
This shuffles the trees with a fixed seed and writes two files.

    python -m src.split_gold --input data/benepar_disagreements.ptb \
        --train-out data/gold_train.ptb --test-out data/gold_test.ptb --test-frac 0.2
"""

import argparse
import random

import nltk


def read_trees(path):
    trees = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                nltk.Tree.fromstring(line)  # validate
            except ValueError:
                print(f"skip un-parseable line: {line[:40]}...")
                continue
            trees.append(line)
    return trees


def main():
    p = argparse.ArgumentParser(description="Split a PTB gold file into train/test.")
    p.add_argument("--input", default="data/benepar_disagreements.ptb")
    p.add_argument("--train-out", default="data/gold_train.ptb")
    p.add_argument("--test-out", default="data/gold_test.ptb")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="Fraction held out for testing (0-1). Default 0.2")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    trees = read_trees(args.input)
    if not trees:
        raise SystemExit(f"No valid trees in {args.input}")

    rng = random.Random(args.seed)
    rng.shuffle(trees)

    n_test = max(1, round(len(trees) * args.test_frac)) if len(trees) > 1 else 0
    test, train = trees[:n_test], trees[n_test:]

    with open(args.train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + ("\n" if train else ""))
    with open(args.test_out, "w", encoding="utf-8") as f:
        f.write("\n".join(test) + ("\n" if test else ""))

    print(f"{len(trees)} trees -> {len(train)} train ({args.train_out}), "
          f"{len(test)} test ({args.test_out}), seed {args.seed}")


if __name__ == "__main__":
    main()

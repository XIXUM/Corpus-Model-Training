#!/usr/bin/env bash
#
# End-to-end retraining + honest evaluation:
#   1. split the gold trees into train / held-out test
#   2. fine-tune Benepar on the train split
#   3. cross-reference the newest checkpoint against the held-out test split
#
# Runs where Benepar + its model are available (i.e. NOT the network-sandboxed
# cloud session — use your local machine, ideally with mps/cuda).
#
# Usage:
#   ./scripts/run_training.sh
#   GOLD=data/benepar_disagreements.ptb TEST_FRAC=0.2 ./scripts/run_training.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GOLD="${GOLD:-data/benepar_disagreements.ptb}"
TRAIN="${TRAIN:-data/gold_train.ptb}"
TEST="${TEST:-data/gold_test.ptb}"
TEST_FRAC="${TEST_FRAC:-0.2}"

# Pick a Python that can import benepar (the training dependency).
pick_python() {
  for c in "${PYTHON:-}" python3.12 python3 python; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import benepar" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}
if ! PYTHON="$(pick_python)"; then
  echo "ERROR: no Python interpreter with 'benepar' was found." >&2
  echo "  Install the project deps:  python3 -m pip install -r requirements.txt" >&2
  echo "  or run with an explicit interpreter:  PYTHON=/path/to/python $0" >&2
  exit 1
fi
echo "==> Using interpreter: $PYTHON"

echo
echo "==> 1/3 Splitting gold set ($GOLD, test fraction $TEST_FRAC)"
"$PYTHON" -m src.split_gold --input "$GOLD" --train-out "$TRAIN" --test-out "$TEST" --test-frac "$TEST_FRAC"

echo
echo "==> 2/3 Training on $TRAIN"
"$PYTHON" -m src.main train --train-data "$TRAIN"

CKPT="$(ls -t checkpoints/benepar_epoch_*.pt 2>/dev/null | head -1 || true)"
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint was produced in checkpoints/." >&2
  exit 1
fi

echo
echo "==> 3/3 Cross-reference on held-out test ($TEST) with $CKPT"
"$PYTHON" -m src.main cross-reference --checkpoint "$CKPT" --test-data "$TEST"

echo
echo "==> Done. Send me the exact-match / F1 / precision / recall numbers above"
echo "    and I'll fill the 'Retraining results' fact sheet."

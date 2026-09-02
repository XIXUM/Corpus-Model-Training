# End-to-end training workflow

The full loop for sustainably improving Benepar: build data → find
disagreements → correct → train → verify → publish. Runs locally, because the
parsing models and training need your hardware.

## 0. Setup (once)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # for the benepar sentencizer
python -c "import benepar; benepar.download('benepar_en3')"
python -c "import nltk; nltk.download('punkt')"
```

Pick your device in `config/training_config.yaml`:

```yaml
hardware:
  device: "mps"   # cuda (NVIDIA) · mps (Apple Silicon) · cpu
```

## 1. Build the corpus

```bash
python -m src.build_gutenberg_corpus --target 500
# -> data/gutenberg_us_corpus.txt  (+ manifest)
```

## 2. Adversarial comparison (find disagreements)

```bash
python -m src.main adversarial \
    --model-a benepar --model-b stanza \
    --data data/gutenberg_us_corpus.txt
```

Outputs:
- `disagreement_logs/…json` — the flagged sentences (input to step 3)
- `tree_logs/…txt` — every Benepar tree
- `reports/latest_comparison_report.html` — visual side-by-side report

> ⚠️ **Gotcha:** adversarial mode **wipes the whole `reports/` directory** at
> the start of each run (`clean_reports`). The committed
> `reports/benepar_corpus_deck.pptx` and any earlier report are deleted locally
> — they are recoverable from git (`git checkout -- reports/…`), but move
> anything you want to keep out of `reports/` before running, or restore it
> after.

## 3. Generate trees for the disagreements

```bash
python -m src.generate_disagreement_trees
# merges into data/benepar_disagreements.ptb (non-destructive; writes a .bak)
```

New sentences are appended; existing corrected trees are preserved (fuzzy
sentence match). Use `--force` only to regenerate everything.

## 4. Correct the trees (the gold step)

Hand-edit the newly appended trees in `data/benepar_disagreements.ptb` so each
is a correct PTB constituency tree. Validate as you go:

```bash
python -c "import nltk; [nltk.Tree.fromstring(l) for l in open('data/benepar_disagreements.ptb') if l.strip()]; print('all trees parse OK')"
```

Record any notable fixes (source typos, recurring Benepar error classes) in
`doc/benepar_error_analysis.md`.

## 5. Train

```bash
python -m src.main train --train-data data/benepar_disagreements.ptb
# checkpoints -> checkpoints/benepar_epoch_N.pt   (epochs/lr in the config)
```

## 5b. One-shot: split + train + held-out eval

For an honest score, evaluate on trees the model was **not** trained on. The
wrapper splits the gold set, trains, and cross-references the new checkpoint
against the held-out split in one go:

```bash
./scripts/run_training.sh                          # 80/20 split of the gold file
GOLD=data/benepar_disagreements.ptb TEST_FRAC=0.2 ./scripts/run_training.sh
```

It picks an interpreter that has `benepar`, writes `data/gold_train.ptb` /
`data/gold_test.ptb` (via `src.split_gold`), trains, then prints exact-match /
F1 / precision / recall on the held-out set. Runs on a machine with the Benepar
model available (not the network-sandboxed cloud session).

## 6. Cross-reference (verify the fixes)

```bash
python -m src.main cross-reference \
    --checkpoint checkpoints/benepar_epoch_5.pt \
    --test-data data/benepar_disagreements.ptb
```

Prints exact-match / F1 / precision / recall and recommends another loop if F1
is below the threshold. For an honest score, verify against a **held-out** gold
file, not the file you trained on.

## 7. Publish

See `doc/huggingface_publish.md` — publish the enlarged corpus (and later a
trained checkpoint) to the Hub.

## Iterating

Repeat steps 1–6 with more/other public-domain sources (raise `--target`, or
add authors in `src/build_gutenberg_corpus.py`). Each pass grows the gold set;
the generator keeps every correction you have already made.

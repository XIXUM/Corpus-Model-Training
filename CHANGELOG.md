# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-09-01

First public iteration: the Benepar gold corpus was cleaned, the tooling made
reproducible and non-destructive, the dataset scaled and published, and the
whole story packaged for presentation.

### Added
- `src/build_gutenberg_corpus.py` — builds a 500-sentence US-English
  public-domain corpus (Melville, Bryant, Burgess via the NLTK Gutenberg
  sample) with a provenance/license manifest.
- `src/split_gold.py` and `scripts/run_training.sh` — split the gold set into a
  train / held-out test split, then train and cross-reference in one call.
- `src/upload_to_huggingface.py` and `scripts/publish_hf.sh` — publish the
  dataset to the HuggingFace Hub (env token or cached `huggingface-cli login`,
  auto-selected interpreter) with a generated dataset card, hero banner, badges
  and a rendered example-tree gallery.
- `src/render_example_trees.py` and `src/render_banner.py` — render the example
  constituency trees and the dataset banner as SVG.
- `doc/benepar_error_analysis.md`, `doc/training_workflow.md`,
  `doc/huggingface_publish.md` — error analysis and runbooks.
- `LICENSE` (MIT for code; Public Domain corpus; CC-BY-4.0 annotations).
- Presentation deck (`reports/benepar_corpus_deck.pptx`).

### Changed
- `generate_disagreement_trees.py` now merges instead of overwriting: existing
  (hand-corrected) trees are preserved via a fuzzy sentence match, a `.bak`
  backup is written each run, and `--force` restores full regeneration.

### Fixed
- Corrected all 38 gold constituency trees in `data/benepar_disagreements.ptb`
  (clause-type mislabels, phrasal-verb particles, coordination attachment, a
  flattened clausal subject, POS consistency, and a documented source typo
  `explain t` → `it`); every tree validates with `nltk.Tree.fromstring`.
- Example-tree SVGs get an opaque white background so they stay legible on dark
  backgrounds (HuggingFace dark mode).
- `clean_reports()` no longer wipes the whole `reports/` directory on an
  adversarial run — it removes only the generated report file(s), preserving the
  deck and `reports/factsheet_assets/`.

[Unreleased]: https://github.com/XIXUM/Corpus-Model-Training/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/XIXUM/Corpus-Model-Training/releases/tag/v0.1.0

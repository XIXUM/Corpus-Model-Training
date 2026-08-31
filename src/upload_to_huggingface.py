"""
Publish the training corpus to the HuggingFace Hub as a `dataset` repo.

What it uploads (by default):

    data/benepar_disagreements.ptb          corrected gold constituency trees (CC-BY-4.0)
    data/gutenberg_us_corpus.txt            US-English public-domain sentences
    data/gutenberg_us_corpus_manifest.json  provenance / license for the above
    README.md                               a generated dataset card (with YAML header)

The source essay `data/ASchoolEssay.txt` is **excluded by default** because it is
personal, non-public-domain text; pass --include-essay to add it explicitly.

Nothing is uploaded without a token. Use --dry-run to assemble the upload folder
and print the dataset card + file list without touching the Hub.

Examples
--------
    # Preview only (no network):
    python -m src.upload_to_huggingface --repo-id xixum/benepar-corpus --dry-run

    # Real upload (needs a write token):
    export HF_TOKEN=hf_xxx
    python -m src.upload_to_huggingface --repo-id xixum/benepar-corpus
"""

import argparse
import json
import os
import shutil
import tempfile
from typing import List, Optional

# Files to publish: (source path, arcname in the repo)
DEFAULT_FILES = [
    ("data/benepar_disagreements.ptb", "data/benepar_disagreements.ptb"),
    ("data/gutenberg_us_corpus.txt", "data/gutenberg_us_corpus.txt"),
    ("data/gutenberg_us_corpus_manifest.json", "data/gutenberg_us_corpus_manifest.json"),
]


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_dataset_card(repo_id: str) -> str:
    gold = _count_lines("data/benepar_disagreements.ptb")
    gutenberg = _count_lines("data/gutenberg_us_corpus.txt")
    pretty = repo_id.split("/")[-1].replace("-", " ").title()

    # YAML metadata header consumed by the Hub.
    header = f"""---
license: cc-by-4.0
language:
- en
tags:
- constituency-parsing
- benepar
- treebank
- penn-treebank
- us-english
task_categories:
- token-classification
pretty_name: {pretty}
size_categories:
- n<1K
---
"""

    body = f"""
# {pretty}

Gold-standard constituency-parsing data for improving the **Benepar**
(`benepar_en3`) US-English parser, plus a larger raw US-English corpus for
generating further training material.

## Contents

| File | Description | Items | License |
|------|-------------|------:|---------|
| `data/benepar_disagreements.ptb` | Manually corrected constituency trees (PTB bracketed format, one tree per line). | {gold} | CC-BY-4.0 |
| `data/gutenberg_us_corpus.txt` | Clean US-English public-domain sentences (one per line). | {gutenberg} | Public Domain |
| `data/gutenberg_us_corpus_manifest.json` | Provenance and license for the corpus above. | — | Public Domain |

## Provenance & licensing

- The **corrected trees** are original annotations, released under
  **CC-BY-4.0**. Each tree was reviewed against its source text and fixed for
  constituent-label, POS, phrasal-verb, coordination and clause-type errors.
- The **Gutenberg corpus** is derived from public-domain works by US-American
  authors (Herman Melville, Sara Cone Bryant, Thornton W. Burgess) via the NLTK
  Gutenberg sample, and is in the **public domain**. See the manifest for the
  exact sources.

## Intended use

Fine-tuning / evaluating US-English constituency parsers. The trees load
directly with `nltk.Tree.fromstring`:

```python
import nltk
with open("data/benepar_disagreements.ptb") as f:
    trees = [nltk.Tree.fromstring(line) for line in f if line.strip()]
```

## Loading the raw corpus

```python
with open("data/gutenberg_us_corpus.txt") as f:
    sentences = [line.strip() for line in f if line.strip()]
```

---
_Generated with the project's `src/upload_to_huggingface.py`._
"""
    return header + body


def assemble(staging: str, files: List, repo_id: str) -> List[str]:
    written = []
    for src, arc in files:
        if not os.path.exists(src):
            print(f"  ! skipping missing file: {src}")
            continue
        dst = os.path.join(staging, arc)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        written.append(arc)
    card_path = os.path.join(staging, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(build_dataset_card(repo_id))
    written.append("README.md")
    return written


def main():
    p = argparse.ArgumentParser(description="Upload the corpus to the HuggingFace Hub.")
    p.add_argument("--repo-id", required=True,
                   help="Target dataset repo, e.g. 'xixum/benepar-corpus'")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace write token (or set HF_TOKEN)")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    p.add_argument("--include-essay", action="store_true",
                   help="Also upload data/ASchoolEssay.txt (personal, non-PD text)")
    p.add_argument("--dry-run", action="store_true",
                   help="Assemble and preview only; do not upload")
    args = p.parse_args()

    files = list(DEFAULT_FILES)
    if args.include_essay:
        files.append(("data/ASchoolEssay.txt", "data/ASchoolEssay.txt"))

    staging = tempfile.mkdtemp(prefix="hf_upload_")
    try:
        written = assemble(staging, files, args.repo_id)
        print(f"Assembled {len(written)} file(s) in {staging}:")
        for w in written:
            print(f"  - {w}")

        if args.dry_run:
            print("\n===== DATASET CARD (README.md) =====\n")
            print(build_dataset_card(args.repo_id))
            print("===== DRY RUN: nothing uploaded =====")
            return

        if not args.token:
            raise SystemExit(
                "No token provided. Set HF_TOKEN or pass --token (or use --dry-run)."
            )

        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise SystemExit(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            )

        api = HfApi(token=args.token)
        print(f"\nCreating/using repo '{args.repo_id}' ({args.repo_type}) ...")
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type,
                        private=args.private, exist_ok=True)
        print("Uploading folder ...")
        api.upload_folder(folder_path=staging, repo_id=args.repo_id,
                          repo_type=args.repo_type,
                          commit_message="Add corrected trees + US-English corpus")
        url = f"https://huggingface.co/datasets/{args.repo_id}"
        print(f"Done: {url}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()

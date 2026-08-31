# Publishing the corpus to the HuggingFace Hub

The dataset is published with `src/upload_to_huggingface.py`. Everything is
already prepared and dry-run verified — you only need a write token and your
Hub namespace.

## What gets published

As a HuggingFace **dataset** repo:

| File | License |
|------|---------|
| `data/benepar_disagreements.ptb` (38 corrected gold trees) | CC-BY-4.0 |
| `data/gutenberg_us_corpus.txt` (500 US-English sentences) | Public Domain |
| `data/gutenberg_us_corpus_manifest.json` | Public Domain |
| `data/ASchoolEssay.txt` (fictional source essay) | CC-BY-4.0 |
| `README.md` (auto-generated dataset card, `en`, `cc-by-4.0`) | — |

Add `--exclude-essay` to leave the essay out.

## Steps

1. Install the client (one-off):

   ```bash
   pip install huggingface_hub
   ```

2. Create a **write** token at <https://huggingface.co/settings/tokens>
   and export it (it is never stored in the repo):

   ```bash
   export HF_TOKEN=hf_xxx
   ```

3. Preview the exact upload folder and dataset card — no network:

   ```bash
   python -m src.upload_to_huggingface --repo-id <namespace>/benepar-corpus --dry-run
   ```

4. Publish (creates the repo if needed, then uploads):

   ```bash
   python -m src.upload_to_huggingface --repo-id <namespace>/benepar-corpus
   ```

   Result: `https://huggingface.co/datasets/<namespace>/benepar-corpus`

Replace `<namespace>` with your user or org (e.g. `xixum`). Add `--private`
to publish privately first.

## Notes

- The token is read from `HF_TOKEN` (or `--token`) at runtime only — do not
  put it in any file, commit, or the dataset card.
- Re-running the command re-uploads changed files to the same repo (a normal
  Hub commit), so it is safe to publish updated corpus versions later.
- A future model checkpoint can be published the same way with
  `--repo-type model` (adjust the card accordingly).

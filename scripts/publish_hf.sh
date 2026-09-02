#!/usr/bin/env bash
#
# Publish the Benepar corpus dataset to the HuggingFace Hub.
#
# Auth (pick one, before calling):
#   export HF_TOKEN=hf_xxxxxxxx      # a HuggingFace *write* token
#   huggingface-cli login           # cached login (token picked up automatically)
#
# Usage:
#   ./scripts/publish_hf.sh                 # -> freshNfunky/benepar-corpus
#   HF_REPO_ID=you/benepar-corpus ./scripts/publish_hf.sh
#   ./scripts/publish_hf.sh --private       # extra flags pass through to the uploader
#
set -euo pipefail

REPO_ID="${HF_REPO_ID:-freshNfunky/benepar-corpus}"

# Run from the repo root regardless of where the script is called from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pick a Python interpreter that actually has huggingface_hub installed
# (on macOS `python3` is often a different interpreter than the `pip` used).
pick_python() {
  for c in "${PYTHON:-}" python3.12 python3 python; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import huggingface_hub" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}
if ! PYTHON="$(pick_python)"; then
  echo "ERROR: no Python interpreter with 'huggingface_hub' was found." >&2
  echo "  Install it for the interpreter you use, e.g.:" >&2
  echo "    python3 -m pip install huggingface_hub" >&2
  echo "  or run this script with an explicit one:  PYTHON=python3.12 $0" >&2
  exit 1
fi
echo "==> Using interpreter: $PYTHON"

# --- Auth check: HF_TOKEN env, or a cached huggingface-cli login -------------
if [ -n "${HF_TOKEN:-}" ]; then
  echo "==> Using HF_TOKEN from the environment."
elif "$PYTHON" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
  echo "==> Using cached 'huggingface-cli login' credentials."
else
  echo "ERROR: no HuggingFace token found." >&2
  echo "  Set one first:  export HF_TOKEN=hf_xxxxxxxx" >&2
  echo "  or log in:      huggingface-cli login" >&2
  exit 1
fi

# --- Make sure the latest committed assets/script are present ---------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git pull --ff-only origin main 2>/dev/null \
    || echo "note: skipped 'git pull' (local changes or offline) — using the current checkout."
fi

# --- Preview, then upload ----------------------------------------------------
echo
echo "==> Dry run (no upload):"
"$PYTHON" -m src.upload_to_huggingface --repo-id "$REPO_ID" "$@" --dry-run

echo
echo "==> Uploading to '$REPO_ID' ..."
"$PYTHON" -m src.upload_to_huggingface --repo-id "$REPO_ID" "$@"

echo
echo "==> Done. Dataset: https://huggingface.co/datasets/$REPO_ID"

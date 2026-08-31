import json
import os
import glob
import sys
import shutil
import argparse
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple

import nltk

# Two sentence keys at or above this character-similarity are treated as the
# same sentence. This lets a manual correction that edited a token (e.g. fixing
# a source typo "explain t" -> "explain it") still be matched back to its
# sentence on a rerun instead of being re-added as a near-duplicate.
_FUZZY_MATCH_THRESHOLD = 0.95

# Add project root to path
sys.path.append(os.getcwd())

from src.models.benepar_wrapper import BeneparWrapper


def find_latest_log(log_dir: str = "disagreement_logs") -> str:
    files = glob.glob(os.path.join(log_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON logs found in {log_dir}")
    return max(files, key=os.path.getctime)


def load_sentences_from_log(log_path: str) -> List[str]:
    print(f"Loading sentences from: {log_path}")
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = [entry['sentence'] for entry in data]
    return sentences


def _sentence_key(leaves: List[str]) -> str:
    """
    Build a tokenization-insensitive key from a list of tree leaves so that a
    hand-corrected tree can be matched back to the same sentence on a rerun.
    Lower-cases and strips everything but alphanumerics, then joins.
    """
    joined = "".join(leaves).lower()
    return "".join(ch for ch in joined if ch.isalnum())


def _match_existing_key(key: str, existing: Dict[str, str]) -> Optional[str]:
    """
    Return the key of an existing tree that represents the same sentence as
    ``key``. Prefers an exact match; otherwise falls back to the most similar
    existing key above ``_FUZZY_MATCH_THRESHOLD`` (so a token-level correction
    still lines up with its original sentence). Returns None if nothing matches.
    """
    if key in existing:
        return key
    best_key: Optional[str] = None
    best_ratio = _FUZZY_MATCH_THRESHOLD
    for candidate in existing:
        # Quick length gate before the more expensive ratio computation.
        if abs(len(candidate) - len(key)) > max(len(key), len(candidate)) * 0.1:
            continue
        ratio = SequenceMatcher(None, key, candidate).ratio()
        if ratio >= best_ratio:
            best_ratio = ratio
            best_key = candidate
    return best_key


def load_existing_trees(path: str) -> Dict[str, str]:
    """
    Load already-present trees from an output file, keyed by their sentence
    (derived from the tree leaves). These may have been manually corrected, so
    they must be preserved across reruns.
    """
    existing: Dict[str, str] = {}
    if not os.path.exists(path):
        return existing
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tree = nltk.Tree.fromstring(line)
                key = _sentence_key(tree.leaves())
            except ValueError:
                # Un-parseable line: keep it under its raw text so it is never lost.
                key = _sentence_key(line.split())
            if key:
                existing[key] = line
    return existing


def generate_trees(sentences: List[str], output_file: str, force: bool = False) -> None:
    """
    Generate PTB trees for the given sentences.

    Unless ``force`` is set, trees already present in ``output_file`` are kept
    verbatim (they may contain manual corrections) and only sentences that are
    not yet represented get a freshly generated Benepar tree. A timestamped
    ``.bak`` copy of the existing file is written before anything is overwritten.
    """
    existing = {} if force else load_existing_trees(output_file)
    if existing:
        print(f"Found {len(existing)} existing tree(s) in {output_file} that will be preserved.")

    # Back up the current file so a rerun can never silently destroy corrections.
    if os.path.exists(output_file):
        backup_path = output_file + ".bak"
        shutil.copy2(output_file, backup_path)
        print(f"Backed up existing file to {backup_path}")

    print("Initializing Benepar model...")
    model = BeneparWrapper(name="benepar", model_name="benepar_en3")

    print(f"Generating trees for {len(sentences)} sentences...")

    preserved = 0
    generated = 0
    failed = 0
    seen_keys = set()

    lines_out: List[str] = []
    for i, sentence in enumerate(sentences):
        tree_str = model.get_tree_string(sentence)
        if not tree_str:
            print(f"Warning: Could not generate tree for sentence: {sentence[:30]}...")
            failed += 1
            continue

        # Key the sentence by the generated tree's leaves (matches how existing
        # trees are keyed), so corrections line up even with odd tokenization.
        try:
            key = _sentence_key(nltk.Tree.fromstring(tree_str).leaves())
        except ValueError:
            key = _sentence_key(sentence.split())

        if key in seen_keys:
            # Duplicate sentence within this log run; skip the repeat.
            continue
        seen_keys.add(key)

        match = _match_existing_key(key, existing)
        if match is not None:
            # Preserve the existing (possibly hand-corrected) tree, and record
            # the matched existing key as seen so it is not re-appended below.
            lines_out.append(existing[match])
            seen_keys.add(match)
            preserved += 1
        else:
            lines_out.append(tree_str)
            generated += 1

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sentences)}")

    # Keep any existing trees whose sentence was not in this log run, so older
    # corrections are never dropped just because the disagreement set changed.
    for key, line in existing.items():
        if key not in seen_keys:
            lines_out.append(line)
            preserved += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_out) + "\n")

    print(
        f"Wrote {len(lines_out)} trees to {output_file} "
        f"({preserved} preserved, {generated} newly generated, {failed} failed)."
    )
    if generated:
        print("You can now manually correct the newly generated trees and use them for training.")
    else:
        print("No new sentences to correct; existing corrections were preserved.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PTB tree file from disagreement logs for manual correction."
    )
    parser.add_argument("--log-dir", type=str, default="disagreement_logs",
                        help="Directory containing disagreement logs")
    parser.add_argument("--output", type=str, default="data/benepar_disagreements.ptb",
                        help="Output file path for trees")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate every tree, discarding manual corrections in the output "
                             "file (a .bak backup is still written).")

    args = parser.parse_args()

    try:
        latest_log = find_latest_log(args.log_dir)
        sentences = load_sentences_from_log(latest_log)
        generate_trees(sentences, args.output, force=args.force)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

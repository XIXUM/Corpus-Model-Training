"""
Build a larger US-English training corpus from public-domain literary texts.

Source: the NLTK Gutenberg sample (a stable, cleaned mirror of Project
Gutenberg texts). Only **US-American authors** are selected, because the target
parser (benepar_en3) is a US-English model:

    - Herman Melville      - Moby Dick (1851)                    - narrative
    - Sara Cone Bryant     - Stories to Tell to Children (1918)  - narrative
    - Thornton W. Burgess  - The Adventures of Buster Bear (1920)- narrative

All three are in the public domain, so the resulting corpus is safe to publish
and to upload to HuggingFace.

The script downloads the texts, strips headers/boilerplate, splits them into
sentences, keeps well-formed sentences in a length band that is realistic to
correct into gold trees, samples a target number (default 500) with a fixed
seed, and writes:

    data/gutenberg_us_corpus.txt            one sentence per line
    data/gutenberg_us_corpus_manifest.json  provenance / license / counts

The output feeds the normal pipeline:  adversarial -> generate_disagreement_trees
-> manual correction -> train.
"""

import argparse
import io
import json
import os
import random
import re
import zipfile
from collections import Counter
from typing import Dict, List

import regex as reg
import requests

GUTENBERG_ZIP_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/corpora/gutenberg.zip"
)

# US-American authors only. Weight tilts toward the simpler narrative registers
# (Bryant, Burgess) that match the project's source essay, with some Melville
# for syntactic variety.
SOURCES: Dict[str, dict] = {
    "gutenberg/bryant-stories.txt": {
        "title": "Stories to Tell to Children",
        "author": "Sara Cone Bryant",
        "year": 1918,
        "weight": 0.40,
    },
    "gutenberg/burgess-busterbrown.txt": {
        "title": "The Adventures of Buster Bear",
        "author": "Thornton W. Burgess",
        "year": 1920,
        "weight": 0.35,
    },
    "gutenberg/melville-moby_dick.txt": {
        "title": "Moby Dick",
        "author": "Herman Melville",
        "year": 1851,
        "weight": 0.25,
    },
}

# Abbreviations that must not trigger a sentence break.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "mt", "capt", "gen", "col", "sgt", "lt",
    "rev", "hon", "prof", "sr", "jr", "vs", "etc", "no", "co", "inc",
}

_HEADER_RE = re.compile(r"^\s*\[.*?\]\s*", re.DOTALL)


def download_texts(url: str) -> Dict[str, str]:
    print(f"Downloading corpus archive from {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    texts = {}
    for name in SOURCES:
        texts[name] = zf.read(name).decode("utf-8", errors="replace")
    print(f"Extracted {len(texts)} source text(s).")
    return texts


def clean_text(raw: str) -> str:
    """Strip the '[Title by Author Year]' header, normalise whitespace, and drop
    obvious structural headings (CHAPTER/VOLUME/roman-numeral lines)."""
    text = _HEADER_RE.sub("", raw, count=1)
    text = text.replace("\r\n", "\n")

    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")  # keep paragraph breaks
            continue
        # Drop all-caps / heading-ish lines (chapter titles, "ETYMOLOGY", etc.).
        letters = [c for c in stripped if c.isalpha()]
        if letters and all(c.isupper() for c in letters) and len(stripped) < 60:
            continue
        if re.match(r"^(chapter|volume|book|part|act|scene)\b", stripped, re.I):
            continue
        if re.match(r"^[IVXLC]+\.?\s*$", stripped):
            continue
        kept_lines.append(stripped)

    # Join lines within a paragraph with spaces; keep blank lines as separators.
    paragraphs = re.split(r"\n\s*\n", "\n".join(kept_lines))
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n".join(p for p in paragraphs if p)


def split_sentences(text: str) -> List[str]:
    """A pragmatic sentence splitter for clean literary prose. The downstream
    models re-tokenise anyway, so this only needs to produce good candidates."""
    sentences: List[str] = []
    for para in text.split("\n"):
        # Break after ., !, ? (plus optional closing quote/paren) when followed
        # by whitespace and a capital letter or opening quote.
        parts = reg.split(r'(?<=[.!?][")\'”’]?)\s+(?=[A-Z"\'“])', para)
        buf = ""
        for part in parts:
            candidate = (buf + " " + part).strip() if buf else part.strip()
            # Merge back if the break followed a known abbreviation.
            last_word = re.findall(r"([A-Za-z]+)\.?$", candidate)
            prev_tok = candidate.split()[-1].rstrip('.').lower() if candidate.split() else ""
            if prev_tok in _ABBREV:
                buf = candidate
                continue
            sentences.append(candidate)
            buf = ""
        if buf:
            sentences.append(buf)
    return sentences


def is_good_sentence(s: str, min_tokens: int, max_tokens: int) -> bool:
    toks = s.split()
    if not (min_tokens <= len(toks) <= max_tokens):
        return False
    if not re.match(r'^["\'“]?[A-Z]', s):
        return False
    if not re.search(r'[.!?]["\')”’]?$', s):
        return False
    if any(ch.isdigit() for ch in s):
        return False
    alpha = sum(c.isalpha() for c in s)
    if alpha < 0.65 * len(s):
        return False
    if "_" in s or s.count('"') % 2 != 0:
        return False
    return True


def build(target: int, min_tokens: int, max_tokens: int, seed: int,
          out_txt: str, out_manifest: str) -> None:
    texts = download_texts(GUTENBERG_ZIP_URL)
    rng = random.Random(seed)

    per_source_pool: Dict[str, List[str]] = {}
    for name, meta in SOURCES.items():
        cleaned = clean_text(texts[name])
        sents = [s for s in split_sentences(cleaned)
                 if is_good_sentence(s, min_tokens, max_tokens)]
        # De-duplicate within a source, preserving order, then shuffle.
        seen = set()
        uniq = []
        for s in sents:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        rng.shuffle(uniq)
        per_source_pool[name] = uniq
        print(f"  {meta['author']:20s}: {len(uniq)} usable sentences")

    # Allocate the target across sources by weight, capped by availability.
    selected: List[str] = []
    provenance: List[dict] = []
    global_seen = set()
    counts = Counter()
    for name, meta in SOURCES.items():
        quota = int(round(target * meta["weight"]))
        taken = 0
        for s in per_source_pool[name]:
            if taken >= quota:
                break
            if s in global_seen:
                continue
            global_seen.add(s)
            selected.append(s)
            provenance.append({"source": name, "author": meta["author"],
                               "title": meta["title"], "text": s})
            counts[name] += 1
            taken += 1

    # Top up (or trim) to hit the target exactly.
    if len(selected) < target:
        leftovers = [s for name in SOURCES for s in per_source_pool[name]
                     if s not in global_seen]
        rng.shuffle(leftovers)
        for s in leftovers:
            if len(selected) >= target:
                break
            selected.append(s)
            provenance.append({"source": "topup", "text": s})
    selected = selected[:target]
    provenance = provenance[:target]

    rng.shuffle(selected)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(selected) + "\n")

    manifest = {
        "description": "US-English public-domain narrative sentences for "
                       "constituency-parser training (Benepar).",
        "license": "Public Domain (Project Gutenberg / NLTK Gutenberg sample)",
        "source_archive": GUTENBERG_ZIP_URL,
        "language": "en-US",
        "seed": seed,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "target": target,
        "total_written": len(selected),
        "counts_by_source": {SOURCES[k]["author"] if k in SOURCES else k: v
                             for k, v in counts.items()},
        "sources": [{"file": k, **{kk: vv for kk, vv in m.items() if kk != "weight"}}
                    for k, m in SOURCES.items()],
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(selected)} sentences to {out_txt}")
    print(f"Wrote provenance/license manifest to {out_manifest}")
    print(f"Per-source counts: {dict(counts)}")


def main():
    p = argparse.ArgumentParser(description="Build a US-English public-domain "
                                            "training corpus from Gutenberg texts.")
    p.add_argument("--target", type=int, default=500, help="Number of sentences to write")
    p.add_argument("--min-tokens", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/gutenberg_us_corpus.txt")
    p.add_argument("--manifest", type=str, default="data/gutenberg_us_corpus_manifest.json")
    args = p.parse_args()
    build(args.target, args.min_tokens, args.max_tokens, args.seed, args.out, args.manifest)


if __name__ == "__main__":
    main()

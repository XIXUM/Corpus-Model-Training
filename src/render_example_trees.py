"""
Render selected gold constituency trees to standalone SVG files.

Used to produce the example trees embedded in the fact sheets. Reads the
corrected corpus, renders a few representative trees (plus a before/after pair
for the "knocked over" phrasal-verb fix) with svgling, and writes them to
reports/factsheet_assets/.

    python -m src.render_example_trees
"""

import os
import subprocess

import nltk
import svgling

CORPUS = "data/benepar_disagreements.ptb"
OUT_DIR = "reports/factsheet_assets"
# Commit that holds the pre-correction trees, for the before/after example.
BEFORE_REV = "64b8094"


def _save(line: str, name: str) -> None:
    tree = nltk.Tree.fromstring(line)
    svg = svgling.draw_tree(tree)._repr_svg_()
    with open(os.path.join(OUT_DIR, name + ".svg"), "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {name}.svg ({len(tree.leaves())} leaves)")


def _first(lines, needle):
    return next(l for l in lines if needle in l)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    current = [l.strip() for l in open(CORPUS, encoding="utf-8") if l.strip()]

    try:
        old_raw = subprocess.run(
            ["git", "show", f"{BEFORE_REV}:{CORPUS}"],
            capture_output=True, text=True, check=True,
        ).stdout
        old = [l.strip() for l in old_raw.split("\n") if l.strip()]
        _save(_first(old, "knocked"), "knocked_before")
    except Exception as e:  # noqa: BLE001 - best-effort historical lookup
        print(f"skip before-tree ({e})")

    _save(_first(current, "knocked"), "knocked_after")
    _save(_first(current, "(S (NP (NNP Mom))"), "mom_forgot")
    _save(_first(current, "(WDT What)"), "what_luck")
    _save(_first(current, "Fascinated"), "fascinated")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()

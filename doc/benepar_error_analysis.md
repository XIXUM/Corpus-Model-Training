# Benepar Error Analysis & Corpus Correction Log

_Last updated: 2026-08-31_

This document records (a) the recurring error classes observed when comparing
the parsing models and (b) the concrete corrections applied to the
gold-standard training corpus `data/benepar_disagreements.ptb`.

## 1. Setup

The adversarial run behind `reports/latest_comparison_report.html`
(generated 2025-12-10) compares:

- **Model A – Benepar** (`benepar_en3`, chart-based transformer) — the **best**
  model in this project.
- **Model B – Stanza** (Stanford constituency) — the **second best**, but
  markedly less precise. Every other wrapper (SuPar/BERT, spaCy-POS) is weaker
  than Stanza.

Across the report there are **49 token-level POS disagreements**. Benepar is
correct in the large majority of them; the tally below is therefore mostly a map
of *Stanza's* weaknesses, with a smaller set of genuine *Benepar* errors that
matter for retraining (Section 3).

## 2. POS disagreement classes (Benepar vs Stanza)

Counts from `reports/latest_comparison_report.html` (`Benepar tag` vs
`Stanza tag`, most frequent first):

| Count | Benepar | Stanza | Example tokens | Who is usually right |
|------:|---------|--------|----------------|----------------------|
| 4 | `IN` | `RB` | so | Benepar (subordinator "so") |
| 4 | `RB` | `JJ` | fast, dear, else, close | mixed / context-dependent |
| 3 | `IN` | `JJ` | about, due | context ("about to", "due to") |
| 3 | `:` | `,` | – (dash) | Benepar (dash is `:` in PTB) |
| 3 | `JJ` | `VBN` | fascinated, embarrassed, hurried | **Benepar** (participles → `VBN`) |
| 2 | `RB` | `NN` | course, t | mixed (see typo note) |
| 2 | `NN`/`NNS` | `IN`/… | by | Benepar (compound "passers-by") |
| 2 | `IN`/`RB` | `RP` | up, over | Stanza (particles → `RP`) in some cases |
| 2 | `NN` | `NNS` | police | **standardized to `NN`** (modifier) |
| 1 | `WP` | `WDT` | what | context (exclamative "what" → `WDT`) |
| 1 | `NNS` | `NNP` | christmas | Benepar/`NNP` for the proper noun |
| 1 | `NNP` | `JJ` | rewe | Benepar (`NNP`, brand name) |
| … | | | | |

**Aggregate tag involvement in disagreements**

- Benepar side: `RB` (10), `IN` (9), `JJ` (7), `:` (4), `NN` (4), `NNS` (3).
- Stanza side: `JJ` (8), `NN` (8), `RB` (6), `IN` (5), `,` (4), `RP` (4), `VBN` (3).

**Dominant confusion families**

1. **Function-word class (RB / IN / RP / JJ)** — the single largest source of
   noise: adverb vs preposition vs particle vs adjective on short words
   (*so, up, over, about, due, close, fast*). This is inherent PTB ambiguity;
   Stanza flips these most.
2. **Participle vs adjective (VBN ↔ JJ)** — *fascinated, embarrassed, hurried,
   loaded*. Predicative/reduced-clause participles should be `VBN`; Stanza
   tends to flatten them to `JJ`.
3. **Punctuation (`:` vs `,`)** — the dash "-" used as a clause separator is `:`
   in PTB; Stanza tags it `,`.
4. **Noun sub-type & compounds (NN/NNS/NNP)** — *police, Christmas, Rewe, Mom* —
   modifier-noun number and proper-noun detection.

## 3. Benepar's own errors (the retraining signal)

These are structural/label mistakes **produced by Benepar itself** in the
generated `benepar_disagreements.ptb`, i.e. the ones worth fixing for retraining.
All were corrected in the gold file (see Section 4).

| Class | Example | Benepar output | Correct |
|-------|---------|----------------|---------|
| **Headline mis-rooted as clause** | "CHRISTMAS ACCIDENT AT THE REWE SUPERMARKET" | `(S (NP …) (VP (NN ACCIDENT) …))` — `NN`-headed VP | `NP` fragment |
| **Clause-type mislabel** | "What luck, …" / "Thank God" / "gracious lady …?" / "Thanks guys" | `SBARQ` / `SINV` | `S` (exclamative / imperative / vocative) |
| **Phrasal-verb particle inside PP** | "knocked over the … cart" | `(PP (PRT (RP over)) (NP …))` | `(PRT (RP over)) (NP …)` |
| **Coordination nested in object NP** | "left the lady … and hurried into …" / "ran … and helped …" | second conjunct trapped inside the object `NP`, or a bare `VBD` conjunct | clause-level `VP` coordination |
| **Clausal subject flattened** | "what was to be expected happened" | free relative left as a bare `SBAR` sibling of the `VP` | `NP`(free relative) subject + `VP` |
| **Tokenization / source typo** | "explain **t**" | `(ADVP (RB t))` | `(NP (PRP it))` — see typo note |

## 4. Corpus correction log (`data/benepar_disagreements.ptb`)

38 trees reviewed against the source text `data/ASchoolEssay.txt`. Corrections
(committed in "Fix(Corpus): Correct constituency trees …"):

- **Tree 1** — headline re-rooted from `S` (with an `NN`-headed `VP`) to an `NP`
  fragment; "CHRISTMAS" attributive → `NNP`.
- **Tree 7** — leading participle "Fascinated" `(ADJP (JJ …))` → `(VP (VBN …))`.
- **Tree 8** — **source typo fixed**: the essay literally reads "I couldn't
  explain **t**, …" (a typo for "it"). Corrected the token to "it" and tagged it
  as the direct object `(NP (PRP it))` instead of `(ADVP (RB t))`. This is the
  only token change; all other trees keep their original tokenization.
- **Tree 10** — "what was to be expected" made the clausal `NP` subject of
  "happened"; "due to" normalized to `(PP (IN due) (PP (TO to) …))`.
- **Tree 11** — "knocked over" fixed to verb + `PRT` particle + object `NP`.
- **Tree 14** — exclamative "What luck, …" `SBARQ`→`S`, "What" `WP`→`WDT`.
- **Tree 16** — imperative "Thank God" `SBARQ`→`S`.
- **Tree 18** — fronted "Pale as a sheet and flustered" made a homogeneous
  `ADJP` coordination.
- **Tree 19** — "everything okay?" split into `SQ` (subject `NP` + predicate
  `ADJP`) instead of a modifier stack.
- **Tree 23** — heterogeneous "Not quite as hurried … **but** with … lights"
  wrapped in `UCP`.
- **Tree 24, 27** — "police officers" modifier noun `NNS`→`NN` (consistency with
  "police radio"/"police car"); "only" moved inside its `VP`.
- **Tree 33** — "Thanks guys" `SINV`→`S` with `NP`(NNS Thanks) + vocative `NP`.
- **Tree 34** — "with shining eyes **and** full of pride" wrapped in `UCP`.
- **Tree 36** — "ran over **and** helped …" fixed to clause-level `VP`
  coordination (second conjunct was a bare `VBD`); "over" `IN`→`RB`.
- **Tree 38** — "left the old lady … **and** hurried into the supermarket" lifted
  out of the object `NP` into proper `VP` coordination.

All 38 trees validated as parseable with `nltk.Tree.fromstring` (the loader used
by `src/training/trainer.py`).

**Policy going forward:** when a source typo or a Benepar mistake is found during
correction, fix it in the tree, keep the tokenization otherwise identical, and
record it here. The generator (`src/generate_disagreement_trees.py`) now
preserves these corrections across reruns (merge + `.bak` backup; `--force` to
override), so this log stays authoritative.

## 5. Recommended priorities

1. **Retrain Benepar** on the corrected 38 trees, focusing the value on the
   Section 3 error classes (clause-type labels, phrasal-verb particles,
   coordination attachment) — these are structural and most impactful.
2. Keep growing the gold set: rerun `adversarial` → `generate_disagreement_trees`
   (now non-destructive) → correct new sentences → `cross-reference`.
3. POS-only confusions from Section 2 that are genuine Stanza errors are *not*
   Benepar's problem and need no corpus change; they mainly confirm Benepar's
   lead over Stanza.

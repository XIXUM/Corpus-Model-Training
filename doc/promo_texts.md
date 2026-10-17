# Promotion texts

Ready-to-post copy for announcing the dataset. Replace links/handles if the
namespace changes. Keep the honest framing (small gold set, method-first) — it
lands far better in these communities than hype.

- Dataset: https://huggingface.co/datasets/freshNfunky/benepar-corpus
- Code: https://github.com/XIXUM/Corpus-Model-Training
- Poster image: `reports/factsheet_assets/social_poster.png`

---

## LinkedIn (German)

We haben einen kuratierten US-englischen Constituency-Parsing-Datensatz
open-source gestellt — inklusive Werkzeugkette und sauberer Lizenzierung. 🌳

Der Hintergrund: Constituency-Parser (die Satzstrukturen als Bäume analysieren)
sind nur so gut wie ihre Trainingsdaten. In unserem Vergleich war **Benepar**
klar das präziseste Modell, **Stanza** nur zweitbestes — aber selbst Benepars
Gold-Korpus hatte systematische Fehler.

Also haben wir aufgeräumt:
✅ 38 Gold-Bäume Satz für Satz gegen die Quelle korrigiert (Phrasal-Verben, Satztypen, Koordination, ein echter Tippfehler inklusive)
✅ Die Fehlerklassen analysiert und dokumentiert — der eigentliche Hebel fürs Retraining
✅ Den Datensatz mit 500 gemeinfreien US-Sätzen (Melville, Bryant, Burgess) erweitert
✅ Alles sauber lizenziert: MIT (Code), Public Domain (Korpus), CC-BY-4.0 (Annotationen)
✅ Eine nicht-destruktive Pipeline gebaut, die manuelle Korrekturen nie wieder überschreibt

Spannend war der methodische Kern: ein adversarialer „Back-to-Back"-Test, der
zwei Modelle gegeneinander laufen lässt und genau die Stellen markiert, an denen
sie uneinig sind. Genau dort sitzen die lohnenden Fehler.

👉 Datensatz auf Hugging Face: huggingface.co/datasets/freshNfunky/benepar-corpus
👉 Code auf GitHub: github.com/XIXUM/Corpus-Model-Training

Nächster Schritt: Retraining auf dem erweiterten Gold-Set und eine ehrliche
Evaluation gegen ein Held-out-Testset. Feedback und Mitstreiter willkommen.

#NLP #MachineLearning #OpenSource #ComputationalLinguistics #HuggingFace #ConstituencyParsing #Benepar #OpenData

---

## LinkedIn (English)

We open-sourced a curated US-English constituency-parsing gold corpus — with the
full tooling and clean licensing. 🌳

Constituency parsers (which analyze sentence structure as trees) are only as good
as their training data. In our evaluation, **Benepar** was clearly the most
precise model and **Stanza** a distant second — yet even Benepar's own gold
corpus contained systematic errors.

So we cleaned it up:
✅ Reviewed and corrected 38 gold trees against the source — phrasal verbs, clause types, coordination attachment, even a genuine source typo
✅ Analyzed and documented the recurring error classes — the real signal for retraining
✅ Grew the dataset with 500 public-domain US sentences (Melville, Bryant, Burgess)
✅ Licensed cleanly: MIT (code), Public Domain (corpus), CC-BY-4.0 (annotations)
✅ Built a non-destructive pipeline so manual corrections are never overwritten again

The methodological core was the fun part: an adversarial "back-to-back" test that
runs two models against each other and flags exactly where they disagree. That's
where the errors worth fixing live.

👉 Dataset on Hugging Face: huggingface.co/datasets/freshNfunky/benepar-corpus
👉 Code on GitHub: github.com/XIXUM/Corpus-Model-Training

Next up: retraining on the enlarged gold set and an honest evaluation against a
held-out test set. Feedback and collaborators welcome.

#NLP #MachineLearning #OpenSource #ComputationalLinguistics #HuggingFace #ConstituencyParsing #Benepar #OpenData

> Posting tip: put the repo/HF link in the **first comment** (LinkedIn throttles
> posts with an external link in the body) and attach the poster PNG as the image.

---

## Hacker News — "Show HN"

**Title** (max 80 chars):

    Show HN: A corrected US-English constituency treebank and the tooling to grow it

**URL:** https://github.com/XIXUM/Corpus-Model-Training

**First comment** (post right after submitting):

I maintain a small constituency-parsing project and kept hitting the same wall:
the parser is only as good as its gold trees, and even the best model's gold
data had systematic mistakes.

So I built a workflow around an adversarial "back-to-back" test: run two
constituency parsers (here Benepar vs Stanza) over the same sentences, log every
token where they disagree, and render the disagreements as side-by-side trees.
That surfaces exactly the spans worth hand-correcting, instead of reviewing
everything.

What's in the repo:
- 38 hand-corrected gold trees (PTB bracket format), reviewed against the source
  text — clause-type mislabels, phrasal-verb particles, coordination attachment,
  a flattened clausal subject, and one real source typo.
- An error analysis separating the parser's own structural errors (the
  retraining signal) from the other model's POS noise.
- A corpus builder that pulls 500 public-domain US-English sentences (Melville,
  Bryant, Burgess via the NLTK Gutenberg sample) with a provenance/license
  manifest.
- A non-destructive tree generator (a rerun used to clobber manual corrections;
  now it merges and backs up), a held-out split + train/eval wrapper, and a
  one-command Hugging Face publisher.

Honest scope: the gold set is still tiny (38 trees) — this is the tooling and
the method, with the corpus just starting to grow. Licensing is clean: MIT
(code), Public Domain (corpus), CC-BY-4.0 (annotations).

Dataset: https://huggingface.co/datasets/freshNfunky/benepar-corpus

Feedback on the adversarial-correction approach and on scaling the gold set
(held-out eval, inter-annotator agreement) very welcome.

> Etiquette: be around to answer comments for an hour or two after posting; no
> vote-begging; post mid-morning US Eastern for best reach.

---

## Reddit — r/LanguageTechnology

**Title:**

    Open US-English constituency treebank + an adversarial "back-to-back" workflow for finding the trees worth correcting

**Body:**

Sharing a small open-source project in case it's useful to others doing
constituency parsing.

**The idea:** a parser is only as good as its gold trees. Instead of reviewing a
whole corpus, I run two parsers over the same sentences (Benepar vs Stanza), log
every token where they disagree, and render those disagreements as side-by-side
trees. The disagreements concentrate the genuinely wrong/ambiguous spans, so
hand-correction effort goes where it matters.

**What's there:**
- 38 hand-corrected gold trees (PTB brackets), fixed against the source:
  SBARQ/SINV → S mislabels, phrasal-verb particles (`knocked over`), coordination
  trapped inside object NPs, a free relative flattened to a bare SBAR, plus a
  documented source typo.
- An error-class write-up that separates Benepar's *structural* errors (worth
  retraining on) from Stanza's POS-tag noise. In our 100-sentence run there were
  49 POS disagreements, dominated by function-word classes (IN/RB/RP/JJ) and
  participle↔adjective (VBN↔JJ).
- A 500-sentence public-domain US-English corpus (Melville/Bryant/Burgess, NLTK
  Gutenberg) with a provenance/license manifest, as raw material for more gold
  trees.
- Non-destructive tree generation (merge + backup), a held-out split and
  train/cross-reference wrapper.

**Caveats:** gold set is still tiny; evaluation on the training file is just a
sanity check, not a real benchmark — the held-out split is there for an honest
score once the set grows. US-English only (matching benepar_en3).

Dataset: https://huggingface.co/datasets/freshNfunky/benepar-corpus
Code: https://github.com/XIXUM/Corpus-Model-Training

Curious whether others use model-disagreement to prioritize annotation, and how
you'd approach inter-annotator agreement at this scale.

---

## Reddit — r/MachineLearning

**Title** (the subreddit requires the `[P]` project flair):

    [P] Using model disagreement to prioritize annotation: a corrected US-English constituency treebank + tooling

**Body:**

Small open-source project on improving constituency-parsing gold data.

Rather than annotating a whole corpus, I run two parsers over the same text
(Benepar vs Stanza) and only hand-correct the trees where they disagree — the
disagreements concentrate the errors worth fixing. Output is 38 corrected PTB
gold trees plus an analysis separating the stronger model's structural errors
(retraining signal) from the weaker model's POS noise.

Also included: a 500-sentence public-domain US-English corpus (with license
manifest) to grow the gold set, a non-destructive tree generator, and a held-out
split + train/eval wrapper.

Scope is honest — gold set is still tiny (38 trees); this is mostly the method
and tooling. MIT / Public Domain / CC-BY-4.0.

Dataset: https://huggingface.co/datasets/freshNfunky/benepar-corpus
Code: https://github.com/XIXUM/Corpus-Model-Training

Would love feedback on the disagreement-driven annotation idea and on honest
evaluation at small scale.

---

## Platform checklist (rollout order)

1. **Zenodo** — mint a DOI for the dataset (citable; can mirror a GitHub release).
2. **Hugging Face** — a community Post, and optionally a small Space (interactive tree demo).
3. **Reddit** — r/LanguageTechnology, then r/MachineLearning (`[P]`), r/datasets.
4. **Hacker News** — Show HN (repo URL + first-comment explainer).
5. **Social** — LinkedIn, Bluesky, Mastodon (fosstodon / sigmoid.social).
6. **Lists** — submit a PR to `awesome-nlp` / `awesome-datasets`.
7. **Academic (optional)** — Papers with Code listing; a short arXiv data note (also unlocks HF Papers); LREC for language resources.

Submit links:
- r/LanguageTechnology: https://www.reddit.com/r/LanguageTechnology/submit?type=TEXT
- r/MachineLearning: https://www.reddit.com/r/MachineLearning/submit?type=TEXT
- Hacker News: https://news.ycombinator.com/submit

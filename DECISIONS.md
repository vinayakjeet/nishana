# Decisions

Every nontrivial choice gets an entry here at the time it is made, not
reconstructed later from memory. Newest entries at the top.

## Format

```
## YYYY-MM-DD: <short title>
**Context:** what problem or question forced a decision.
**Decision:** what was chosen.
**Alternatives considered:** what else was on the table, and why it lost.
**Consequences:** what this makes easier or harder later.
```

## 2026-08-24: The eval set is its own corpus, not a copy of ShipGate's

**Context:** ShipGate already carries a 100-item support-intent dataset in the
same domain, and reusing it would save a day of writing. Nishana's brief says
the labeling effort is shared, not the artifact.

**Decision:** author a separate 120-item eval set with different intents
(order_status, refund, account, other rather than billing, technical, account,
other), entity annotations, and a heavier Hinglish/Devanagari mix. Run the same
n-gram decontamination between seeds and eval that the synthetic pool gets.

**Alternatives considered:** importing ShipGate's dataset, which would couple
two projects' baselines to one file and make both numbers move when either repo
edits it; and copying its items with light paraphrase, which is the worst of
both: near-duplicate content with no shared versioning discipline. Also, a
dataset whose items exist in another public repo cannot demonstrate
decontamination tooling honestly, because contamination against itself is
guaranteed.

**Consequences:** a day of authoring spent; two datasets that can gate each
other's claims independently; entity extraction now has gold labels, which
ShipGate's intent-only set never needed. If the portfolio later wants one
merged support-ticket benchmark, the n-gram overlap check runs between them
first.

## 2026-08-24: Blind judged quality is pairwise, both orders, anonymous slots

**Context:** the ablation needs a quality metric that can disagree with
similarity metrics, or it will miss the exact failure mode the honesty
reference documents. Any judge brings position bias; ShipGate measured a judge
whose run-to-run spread was 20 points on identical inputs.

**Decision:** blind pairwise comparison. The judge sees ticket plus two answers
as A and B; variant names, model strings, and provenance never enter the
prompt. Slot order per item comes from a hash of the item id so neither input
file colonises slot A across the corpus, and every pair is judged twice with
candidates swapped. Only consistent winners count toward a variant's quality
score; the swap-disagreement rate publishes beside every judged number.

**Alternatives considered:** single-pass pointwise scoring (cheaper, but
inherits whatever scale calibration the judge happens to have); judging with
variant labels visible (measures brand preference); averaging swapped orders
into one number (hides the disagreement rate that tells you how much to trust
the average).

**Consequences:** judge quota doubles versus single-pass; free tiers sustain it
at the throttled rates the chassis client already enforces. Judged columns stay
null in the ablation table until this script has run for a pair, and null prints
as a gap rather than zero, so missing measurement never masquerades as a tie.

## 2026-08-24: Adapter-only checkpointing, optimizer state deliberately dropped

**Context:** Kaggle sessions end without warning at a wall-clock cap, and the
next session may land on a T4 after starting on a P100. Full trainer state is
gigabytes and does not survive the platform boundary cleanly.

**Decision:** checkpoint the adapter weights plus a manifest (base model,
revision, rank, variant, dataset hash, step range, session number, adapter
digest). Resume validates all of it and refuses any mismatch. Optimizer and
scheduler state do not transfer; the loss jump at each session boundary is
printed from the training logs.

**Alternatives considered:** full checkpoint upload to Kaggle datasets (size
limits and friction make multi-session work fragile), and resuming without
validation (a rank-16 adapter loaded onto a rank-64 config or a changed dataset
silently produces a model that trains without learning what the manifest
claims).

**Consequences:** two-session runs become routine rather than heroic, matching
the documented precedent of arXiv 2504.15610 (loss 1.01 to 0.34 across P100 to
T4 with adapter-only transfer). Each boundary costs some optimizer momentum;
that cost is visible in logs instead of buried. The manifest digest also gives
every prediction row a checksum-shaped identity: `variant@hash8`.

## 2026-08-24: Decontamination reports numbers and offender ids, not assurances

**Context:** "we ran decontamination" is unverifiable prose. The brief demands
n-gram overlap be published, not claimed.

**Decision:** scripts/decontaminate.py measures 5-gram and 8-gram overlap
between seed texts and eval items, writes results/decontam.json with counts,
percentages, and per-item offender lists, and offers --apply to drop offending
items and re-version the dataset. Report-only is the default so removal is
always a recorded decision rather than a silent side effect.

**Alternatives considered:** embedding-similarity decontamination (better
recall for paraphrases, but adds a model dependency to the one pipeline that
must stay offline-testable, and its thresholds are as judgement-laden as the
n-gram lengths); asserting cleanliness without artifacts, which is exactly how
contaminated benchmarks keep publishing state-of-the-art numbers.

**Consequences:** boilerplate phrases can trip 5-gram checks (politeness frames
repeat legitimately in support text). The tool surfaces them with ids attached
so a human inspects and documents rather than silently deleting data; the first
real finding was a tokenizer bug, caught because the report printed the actual
shared n-grams.

## 2026-08-24: No web service in this repo

**Context:** the chassis ships with a FastAPI app, Render deploy, Docker
compose, and Postgres. Every prior project kept some of it.

**Decision:** remove app/, render.yaml, docker-compose.yml, and the Dockerfile.
Serving is deploy/modal_serve.py on Modal behind an OpenAI-compatible surface,
and evaluation speaks ShipGate's adopting contract over HTTP. Nothing in this
repo needs a database.

**Alternatives considered:** keeping a dashboard service for run history
(premature: there are no runs until Kaggle produces adapters, and ShipGate
already renders run history publicly); keeping the FastAPI shell warm for
later (dead code ships in the public repo and rots).

**Consequences:** CI is ruff, conventions, pytest only; deploys are one modal
command once an adapter exists. Tollgate registers the served URL as a
specialist route when the endpoint answers, which is integration by contract
rather than by shared code.

## 2026-08-24: ROUGE-L first, BERTScore pluggable, never substituted silently

**Context:** the similarity column should reward matching the training
distribution, which is why the honesty reference blames it for masking quality
regressions. BERTScore is the published instrument; it pulls a transformer and
does not belong in the offline test path.

**Decision:** implement ROUGE-L in-repo as the default similarity metric.
Expose bertscore_similarity() returning None when the package is absent;
callers print a gap, never substitute the lexical score under the same column.

**Alternatives considered:** requiring bert-score as a hard dependency (breaks
the five-minute offline setup rule and CI determinism); reporting BLEU (worse
behaved at sentence length than LCS F-measure).

**Consequences:** the fidelity column is always populated and comparable across
variants; if BERTScore lands later it appears as its own named column, so the
lexical and model-based numbers can disagree in public.

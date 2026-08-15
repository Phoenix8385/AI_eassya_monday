"""Generate backend/EVALUATION.md from the trained model and cached features.

    python -m scripts.generate_evaluation

Reuses what earlier phases already produced -- the saved classifier
(``model`` + ``scaler`` + ``feature_names``) and ``app/data/features_cache.csv``
-- so no essay is re-scored through GPT-2 here.

**Manual sections survive regeneration.** Two parts of the report have to be
written by a human who has read the evidence: the explanation for each confident
failure, and the ESL-bias reasoning. Those live between ``MANUAL`` markers; on
every run this script reads the existing EVALUATION.md, lifts whatever is inside
those markers, and puts it back. Re-running after a retrain therefore refreshes
every number without destroying the reasoning around them.

The report's own language follows the same rule as the API: probabilities are
framed as resemblance to training examples, never as verdicts, and never as a
bare percentage standing on its own.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT, get_settings  # noqa: E402
from app.services.classifier import confidence_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("generate_evaluation")

DATA_DIR = BACKEND_ROOT / "app" / "data"
DEFAULT_CACHE = DATA_DIR / "features_cache.csv"
DEFAULT_OUT = BACKEND_ROOT / "EVALUATION.md"
EXCERPT_COLUMN = "_text_excerpt"

FAILURE_PROMPT = (
    "[Explain: does this look like a formulaic human essay, or an unusually "
    "clean AI/polished one? Read the excerpt above before answering.]"
)

# Pre-filled rather than left as brackets: this reasoning is about the feature
# set and known characteristics of L2 writing, neither of which needs the
# dataset to answer. Revise it if the feature set changes.
ESL_DEFAULT = """This dataset does not label essays by writer's first-language
background, so this cannot be tested directly.

**A proper test would require** either a labelled ESL subset — essays with the
writer's L1 recorded, ideally alongside proficiency level — or, failing that, a
held-out set annotated for known L2-English markers (article omission or
overuse, preposition substitution, limited idiom, restricted subordination) so
accuracy could be compared across essays with and without them. Either design
needs the annotation to be independent of the label, or the test measures the
annotator's assumptions rather than the model's behaviour. Neither exists here,
so the section below is reasoning about mechanism, not a measurement.

**Based on the features used, the theoretical risk is** concrete and runs in
one direction. Three of the strongest signals are the ones most likely to
misfire on L2 writing:

- **Perplexity.** GPT-2 finds conventional, high-frequency phrasing predictable.
  L2 writers often work within a smaller, safer set of constructions, which
  produces exactly the low-perplexity profile the model reads as machine-like.
- **Burstiness.** The detector treats varied sentence rhythm as human. Writers
  less confident in a second language tend toward more uniform sentence length
  and simpler subordination — low burstiness, which pushes toward the AI side.
- **Transition-word rate.** Explicit connectives (`however`, `moreover`,
  `furthermore`) are taught heavily in EAP and ESL instruction and appear more
  densely in that writing. The model learned to associate them with AI text.

Each of these is a case where the model's evidence is not "this was generated"
but "this is regular", and regularity has causes other than authorship. The
plausible failure is therefore a systematically higher false-positive rate for
non-native speakers — the group least able to contest the result.

This is unmeasured, not disproven, and the direction of the risk is predictable
enough that it should be treated as likely until tested. It is one of the
reasons the API reports resemblance with a confidence sentence rather than a
verdict, and why per-sentence flags are gated behind an essay-level result."""


def read_manual_blocks(path: Path) -> dict[str, str]:
    """Lift human-written sections out of an existing report."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"<!-- MANUAL:(?P<key>[\w.-]+) -->\n(?P<body>.*?)\n<!-- /MANUAL:(?P=key) -->",
        re.DOTALL,
    )
    blocks = {m.group("key"): m.group("body") for m in pattern.finditer(text)}
    if blocks:
        logger.info("Preserving %d manual section(s): %s", len(blocks), sorted(blocks))
    return blocks


def manual(key: str, preserved: dict[str, str], default: str) -> str:
    """Emit a manual block, keeping any previously written content."""
    body = preserved.get(key, default)
    return f"<!-- MANUAL:{key} -->\n{body}\n<!-- /MANUAL:{key} -->"


def format_confusion(y_true, y_pred) -> str:
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return (
        "```\n"
        "                        predicted\n"
        "                     human      AI\n"
        f"    actual human  {tn:>7} {fp:>7}\n"
        f"           AI     {fn:>7} {tp:>7}\n"
        "```\n\n"
        f"- **{fp}** false positive(s): a human-written essay scored as AI-leaning. "
        "This is the error that matters most — it points at a real student.\n"
        f"- **{fn}** false negative(s): an AI-assisted essay scored as human-leaning."
    )


def load_inputs(model_path: Path, cache_path: Path):
    """Load the saved classifier and the cached feature matrix."""
    missing = [p for p in (model_path, cache_path) if not p.exists()]
    if missing:
        lines = ["Cannot generate EVALUATION.md - missing input(s):"]
        lines += [f"  {p}" for p in missing]
        lines += ["", "Build them first:", "  python -m scripts.train_classifier", ""]
        lines.append("Refusing to write a report with invented numbers.")
        raise SystemExit("\n".join(lines))

    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{model_path} is a bare estimator; expected the training dict.")

    model = payload.get("model") or payload.get("estimator")
    feature_names = list(payload.get("feature_names") or [])
    if model is None or not feature_names:
        raise SystemExit(f"{model_path} lacks 'model' or 'feature_names'.")

    cache = pd.read_csv(cache_path)
    absent = (set(feature_names) | {"label", "source"}) - set(cache.columns)
    if absent:
        raise SystemExit(
            f"{cache_path} is missing {sorted(absent)} - it predates the current "
            "feature set. Re-run: python -m scripts.train_classifier --refresh-cache"
        )

    if EXCERPT_COLUMN not in cache.columns:
        logger.warning(
            "Cache has no %s column, so failures cannot be quoted. Re-run "
            "`python -m scripts.train_classifier --refresh-cache` first.",
            EXCERPT_COLUMN,
        )

    return payload, model, feature_names, cache


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=settings.resolved_model_path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-failures", type=int, default=3)
    args = parser.parse_args()

    payload, model, feature_names, cache = load_inputs(args.model, args.cache)

    X = cache[feature_names].to_numpy(dtype=float)
    y = cache["label"].to_numpy(dtype=int)
    sources = cache["source"].astype(str).to_numpy()
    excerpts = (
        cache[EXCERPT_COLUMN].astype(str).to_numpy()
        if EXCERPT_COLUMN in cache.columns
        else np.array(["(excerpt unavailable - re-run with --refresh-cache)"] * len(cache))
    )
    indices = np.arange(len(cache))

    if len(np.unique(y)) < 2:
        raise SystemExit("Cache contains a single class; nothing to evaluate.")

    # --- Held-out split, reproducing train_classifier.py's split exactly ----
    stratify = y if np.bincount(y).min() >= 2 else None
    idx_train, idx_test = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )
    fitted = clone(model).fit(X[idx_train], y[idx_train])
    y_test = y[idx_test]
    test_proba = fitted.predict_proba(X[idx_test])[:, 1]
    test_pred = fitted.predict(X[idx_test])

    # --- Cross-validated metrics ------------------------------------------
    folds = min(args.folds, max(int(np.bincount(y).min()), 2))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    cv_accuracy = cross_val_score(clone(model), X, y, cv=cv, scoring="accuracy")
    oof_pred = cross_val_predict(clone(model), X, y, cv=cv)

    # --- Per-source accuracy, from out-of-fold predictions ----------------
    per_source = {}
    for source in sorted(set(sources)):
        mask = sources == source
        per_source[source] = {
            "n": int(mask.sum()),
            "correct": int((oof_pred[mask] == y[mask]).sum()),
            "accuracy": float((oof_pred[mask] == y[mask]).mean()),
            "labels": sorted(set(y[mask].tolist())),
        }

    # --- Highest-confidence wrong predictions on the test split -----------
    wrong = [i for i in range(len(idx_test)) if test_pred[i] != y_test[i]]
    # Confidence = distance from the decision boundary, so a wrong "human"
    # call at p=0.02 ranks alongside a wrong "AI" call at p=0.98.
    wrong.sort(key=lambda i: abs(test_proba[i] - 0.5), reverse=True)
    failures = []
    for i in wrong[: args.top_failures]:
        row = idx_test[i]
        failures.append(
            {
                "excerpt": " ".join(str(excerpts[row]).split())[:150],
                "source": sources[row],
                "true_label": "human" if y_test[i] == 0 else "AI",
                "probability": float(test_proba[i]),
            }
        )

    preserved = read_manual_blocks(args.out)
    report = render(
        payload=payload,
        cache=cache,
        y=y,
        y_test=y_test,
        test_pred=test_pred,
        oof_pred=oof_pred,
        cv_accuracy=cv_accuracy,
        folds=folds,
        per_source=per_source,
        failures=failures,
        n_wrong=len(wrong),
        n_test=len(idx_test),
        n_train=len(idx_train),
        settings=settings,
        preserved=preserved,
        args=args,
    )

    args.out.write_text(report, encoding="utf-8")
    logger.info(
        "Test split: %d rows, %d wrong. CV accuracy %.4f +/- %.4f over %d folds.",
        len(idx_test),
        len(wrong),
        cv_accuracy.mean(),
        cv_accuracy.std(),
        folds,
    )
    logger.info("Wrote %s", args.out)


def render(**k) -> str:
    payload, settings, preserved, args = k["payload"], k["settings"], k["preserved"], k["args"]
    per_source, failures = k["per_source"], k["failures"]
    y, cv = k["y"], k["cv_accuracy"]

    n_human, n_ai = int((y == 0).sum()), int((y == 1).sum())
    total = n_human + n_ai
    baseline = 100.0 * max(n_human, n_ai) / total

    # --- Accuracy by source, and the gap ---------------------------------
    daigt = next((v for s, v in per_source.items() if s.startswith("daigt")), None)
    polished = next((v for s, v in per_source.items() if "polish" in s), None)

    source_rows = "\n".join(
        f"| `{s}` | {v['n']} | {v['correct']}/{v['n']} | {v['accuracy']:.4f} |"
        for s, v in sorted(per_source.items())
    )

    if daigt and polished:
        gap = daigt["accuracy"] - polished["accuracy"]
        if gap > 0:
            gap_section = f"""### Finding: the AI-polished essays are markedly harder

The gap is **{gap:.4f}** ({gap * 100:.1f} percentage points): the classifier is
right on {daigt['accuracy']:.1%} of the DAIGT essays and {polished['accuracy']:.1%}
of the AI-polished ones.

This is the headline result of the evaluation, not a caveat. The polished set is
human writing with only its surface machine-touched — the ideas, structure and
voice are the author's. A detector trained largely on wholly-generated text
learns the statistical signature of generated *prose*, and lightly-edited human
work does not carry it. The blended figure hides this completely, because the
DAIGT base is {daigt['n']} essays against the polished set's {polished['n']}.

Read it as the honest bound on the tool: it identifies machine-written essays far
better than it identifies machine-*assisted* ones, and machine-assisted is the
case most likely to arrive in real admissions reading."""
        else:
            gap_section = f"""### Finding: the AI-polished essays are not harder here

The polished set scored {polished['accuracy']:.1%} against DAIGT's
{daigt['accuracy']:.1%} — a gap of {gap:.4f}, which is not in the expected
direction.

Treat this with suspicion rather than satisfaction. With {polished['n']} polished
essays, a handful of rows moving swings the figure several points, and the
polished essays are rewrites of essays whose originals may sit in the training
folds. Check for that leakage before reporting this as a positive result."""
    else:
        gap_section = """### Finding: not computable

The cache does not contain both a `daigt` and an `ai_polished` source, so the
comparison that matters most cannot be made. Build both datasets and retrain:

```bash
python -m scripts.prepare_daigt
python -m scripts.build_polished_set
python -m scripts.train_classifier --refresh-cache
```"""

    # --- Confident failures ----------------------------------------------
    if failures:
        blocks = []
        for n, f in enumerate(failures, start=1):
            blocks.append(
                f"""#### Failure {n} — true label: **{f['true_label']}**, source: `{f['source']}`

> {f['excerpt']}...

The classifier placed this essay at **{f['probability']:.4f}** on the AI axis, which
the API would report as:

> {confidence_label(f['probability'])}

That reading is wrong: this essay is **{f['true_label']}**-written.

{manual(f"failure.{n}", preserved, FAILURE_PROMPT)}"""
            )
        failure_section = "\n\n".join(blocks)
        if k["n_wrong"] < args.top_failures:
            failure_section = (
                f"Only {k['n_wrong']} misclassification(s) exist in the "
                f"{k['n_test']}-row test split, so fewer than "
                f"{args.top_failures} are shown.\n\n" + failure_section
            )
    else:
        failure_section = (
            f"No misclassifications in the {k['n_test']}-row held-out split. On a "
            "split this small that is far more likely to mean the split is too "
            "small to contain a hard case than that the model is flawless — "
            "check the cross-validated figure above before reading anything into it."
        )

    return f"""# Evaluation

<!-- GENERATED FILE. Regenerate: python -m scripts.generate_evaluation
     Numbers are computed from the saved model and app/data/features_cache.csv.
     Text between <!-- MANUAL:... --> markers is written by hand and is
     PRESERVED across regenerations - edit it directly. -->

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from model
version `{payload.get('version', 'unknown')}` over {total} cached essays
({n_human} human, {n_ai} AI).

Throughout: probabilities describe **resemblance to the training examples**, not
a verdict about a person. A figure never appears here as a bare percentage
presented as a judgement, for the same reason the API never returns one.

## Test-Set Metrics

A single held-out split — {k['n_train']} essays trained on, {k['n_test']} held
back ({args.test_size:.0%}, `random_state={args.seed}`). **This is the weaker of
the two numbers in this report.** One split's accuracy moves on which essays
happened to land in the test fold; treat it as a spot check and the
cross-validated figure below as the result.

```
{classification_report(k['y_test'], k['test_pred'], target_names=['human', 'AI'], zero_division=0)}
```

## Cross-Validated Metrics

{k['folds']}-fold stratified cross-validation over all {total} essays. Every
essay is predicted exactly once, while held out. **This is the number to quote.**

| Metric | Value |
|---|---|
| Accuracy (mean ± sd) | **{cv.mean():.4f} ± {cv.std():.4f}** |
| Per-fold | {', '.join(f'{a:.4f}' for a in cv)} |
| Majority-class baseline | {baseline:.1f}% |

The baseline row is the one to check first: a model that ignored its input and
always answered `{'human' if n_human >= n_ai else 'AI'}` would score
{baseline:.1f}% on this class balance. An accuracy near that figure means
nothing was learned, however respectable it looks.

```
{classification_report(k['y'], k['oof_pred'], target_names=['human', 'AI'], zero_division=0)}
```

## Confusion Matrix

Pooled out-of-fold predictions — every essay counted once, on a fold where it
was held out.

{format_confusion(k['y'], k['oof_pred'])}

## Accuracy by Source (DAIGT vs AI-Polished)

| Source | Essays | Correct | Accuracy |
|---|---:|---:|---:|
{source_rows}

{gap_section}

## 3 Confident Failures

The misclassifications the model was **most sure about** — ranked by distance
from the decision boundary, so a confidently-wrong "human" call ranks alongside
a confidently-wrong "AI" one. These are the informative errors: a near-50%
mistake is the model admitting uncertainty, while these are not.

{failure_section}

## ESL-Bias Discussion

{manual("esl", preserved, ESL_DEFAULT)}

---

## Appendix: per-sentence flagging thresholds

Design decisions rather than measurements, kept here so the thresholds in the
API are documented alongside the numbers they interact with.

A sentence is flagged only when **all three** gates open:

| Gate | Setting | Default | What it does |
|---|---|---:|---|
| Essay-level probability | `AI_DECISION_THRESHOLD` | `{settings.ai_decision_threshold}` | The essay itself must score AI-leaning |
| Combined weighted deviation | `SENTENCE_FLAG_Z` | `{settings.sentence_flag_z}` | How far the sentence stands out from its own essay |
| Dominant signal's raw z-score | `SENTENCE_MIN_Z` | `{settings.sentence_min_z}` | Floor before any reason template may fire |

**Why the essay-level gate.** Every essay contains its own longest and its own
most predictable sentence — that is what a distribution is. Highlighting one
inside an essay scored at p(AI)=0.05 tells the reader nothing true. This was a
real defect before the gate existed: a human essay scoring 0.052 still produced
a flagged sentence at 1.75σ. The number was right and the flag was meaningless.

**Why an absolute threshold, not a percentile.** A top-N% rule always flags
something — feed it a wholly human essay and it still highlights its worst
quarter, in the same language it uses for machine-written text. An absolute
threshold can return zero flags, and "nothing stood out" is a common correct
answer.

**Why {settings.min_sentences_for_localization} sentences minimum.** Below
`MIN_SENTENCES_FOR_LOCALIZATION`, the essay's own standard deviation comes from
two or three values and is dominated by whichever sentence is longest. Those
essays return `local_score: null` with an explicit reason rather than a
fabricated score.

**Per-sentence flags are unevaluated.** Neither dataset carries per-sentence
ground truth — labels are per-essay, so which sentences an AI touched is
unknown even in the polished subset. The flags are explanatory (which sentences
drove this score), not evidential (which sentences were machine-written), and
the API's wording keeps to resemblance for exactly that reason.

## Scope

These numbers are bounded by what the dataset contains. See `DATASET.md` —
single-language, narrow in genre, no label for writer background, one generation
of AI models, one depth of AI editing.
"""


if __name__ == "__main__":
    main()

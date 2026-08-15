"""Generate backend/DATASET.md from the actual dataset CSVs.

    python -m scripts.write_dataset_doc

Every number in DATASET.md is computed here from the real files. Nothing is
typed by hand, so the document cannot drift out of sync with the data: re-run
this after any change to the dataset and the counts update.

Prerequisites (both CSVs must already exist):

    python -m scripts.prepare_daigt         -> app/data/daigt_essays.csv
    python -m scripts.build_polished_set    -> app/data/ai_polished_essays.csv

The prose is authored; the counts, percentages, ratios and word statistics are
derived. If you edit the narrative, edit the templates below rather than the
generated DATASET.md, or the next run will overwrite your changes.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("write_dataset_doc")

DATA_DIR = BACKEND_ROOT / "app" / "data"
DEFAULT_DAIGT = DATA_DIR / "daigt_essays.csv"
DEFAULT_POLISHED = DATA_DIR / "ai_polished_essays.csv"
DEFAULT_RAW_TRAIN = DATA_DIR / "raw" / "train_essays.csv"
DEFAULT_OUT = BACKEND_ROOT / "DATASET.md"

_WORD_RE = re.compile(r"\S+")


def word_stats(series: pd.Series) -> dict[str, float]:
    """Word-count distribution for a column of essay text."""
    counts = series.astype(str).map(lambda t: len(_WORD_RE.findall(t)))
    return {
        "min": int(counts.min()),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "max": int(counts.max()),
    }


def collect(daigt_path: Path, polished_path: Path, raw_train: Path) -> dict:
    """Read both CSVs and compute every figure the document needs."""
    missing = [p for p in (daigt_path, polished_path) if not p.exists()]
    if missing:
        lines = ["Cannot write DATASET.md - missing input file(s):"]
        for path in missing:
            lines.append(f"  {path}")
        lines.append("")
        lines.append("Build them first:")
        if not daigt_path.exists():
            lines.append("  kaggle competitions download -c llm-detect-ai-generated-text")
            lines.append("  unzip llm-detect-ai-generated-text.zip -d app/data/raw")
            lines.append("  python -m scripts.prepare_daigt")
        if not polished_path.exists():
            lines.append("  python -m scripts.build_polished_set")
        lines.append("")
        lines.append(
            "Refusing to generate a document with invented numbers."
        )
        raise SystemExit("\n".join(lines))

    daigt = pd.read_csv(daigt_path)
    polished = pd.read_csv(polished_path)

    for frame, path, required in (
        (daigt, daigt_path, {"text", "label"}),
        (polished, polished_path, {"text", "original_text", "label", "prompt_variant"}),
    ):
        absent = required - set(frame.columns)
        if absent:
            raise SystemExit(
                f"{path.name} is missing column(s) {sorted(absent)}; found "
                f"{list(frame.columns)}."
            )

    daigt_human = int((daigt["label"] == 0).sum())
    daigt_ai = int((daigt["label"] == 1).sum())
    n_polished = int(len(polished))

    total = int(len(daigt)) + n_polished
    ai_total = daigt_ai + n_polished

    stats: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "daigt_path": daigt_path.name,
        "polished_path": polished_path.name,
        "daigt_rows": int(len(daigt)),
        "daigt_human": daigt_human,
        "daigt_ai": daigt_ai,
        "polished": n_polished,
        "total": total,
        "ai_total": ai_total,
        "pct_human": 100.0 * daigt_human / total if total else 0.0,
        "pct_ai": 100.0 * ai_total / total if total else 0.0,
        "pct_polished_of_total": 100.0 * n_polished / total if total else 0.0,
        "pct_polished_of_ai": 100.0 * n_polished / ai_total if ai_total else 0.0,
        "human_words": word_stats(daigt.loc[daigt["label"] == 0, "text"]),
        "polished_words": word_stats(polished["text"]),
        "original_words": word_stats(polished["original_text"]),
        "variants": polished["prompt_variant"].value_counts().sort_index().to_dict(),
    }

    if daigt_ai:
        stats["raw_ai_words"] = word_stats(daigt.loc[daigt["label"] == 1, "text"])

    # Imbalance
    larger, smaller = max(daigt_human, ai_total), min(daigt_human, ai_total)
    stats["majority_class"] = "human" if daigt_human >= ai_total else "AI"
    stats["minority_class"] = "AI" if daigt_human >= ai_total else "human"
    stats["imbalance_ratio"] = (larger / smaller) if smaller else float("inf")
    stats["majority_baseline"] = 100.0 * larger / total if total else 0.0
    stats["is_skewed"] = stats["imbalance_ratio"] >= 1.5

    # How many polished originals are still present in the human set. The
    # polished essays are derived from human essays that are themselves counted
    # in the human total, so the two are related, not independent samples.
    human_texts = set(daigt.loc[daigt["label"] == 0, "text"].astype(str).str.strip())
    stats["polished_originals_in_human_set"] = int(
        polished["original_text"].astype(str).str.strip().isin(human_texts).sum()
    )

    # Optional enrichment: the normalized CSV drops prompt_id, so read the raw
    # competition file if it is still around.
    stats["distinct_prompts"] = None
    if raw_train.exists():
        try:
            raw = pd.read_csv(raw_train, usecols=["prompt_id"])
            stats["distinct_prompts"] = int(raw["prompt_id"].nunique())
        except (ValueError, KeyError):
            pass

    return stats


def render(s: dict) -> str:
    """Build the Markdown. Prose is authored; every figure comes from `s`."""
    hw, pw, ow = s["human_words"], s["polished_words"], s["original_words"]

    if s["is_skewed"]:
        balance_note = (
            f"**This is skewed.** The split is roughly "
            f"{s['imbalance_ratio']:.1f}:1 in favour of {s['majority_class']} "
            f"essays. A classifier that ignores its input entirely and always "
            f"predicts \"{s['majority_class']}\" scores "
            f"**{s['majority_baseline']:.1f}% accuracy** on this distribution. "
            f"Any accuracy figure in EVALUATION.md at or below that number is "
            f"worthless, and figures modestly above it are still weak evidence — "
            f"read per-class precision and recall for the "
            f"{s['minority_class']} class, and ROC-AUC, instead. Training uses "
            f"`class_weight=\"balanced\"` for the same reason."
        )
    else:
        balance_note = (
            f"The classes are close to balanced ({s['imbalance_ratio']:.2f}:1), "
            f"so overall accuracy is a reasonable headline metric here — though "
            f"EVALUATION.md still reports per-class precision and recall, "
            f"because the two error directions carry very different costs when "
            f"the subject is a student's essay."
        )

    prompt_line = (
        f"The competition's raw training file spans "
        f"**{s['distinct_prompts']} distinct prompts**."
        if s["distinct_prompts"] is not None
        else (
            "The normalized CSV keeps only `text` and `label`, so the prompt "
            "count is not recoverable from it; re-read "
            "`app/data/raw/train_essays.csv` if you need that breakdown."
        )
    )

    variant_rows = "\n".join(
        f"| `{name}` | {count} |" for name, count in s["variants"].items()
    )

    raw_ai_row = ""
    if "raw_ai_words" in s:
        r = s["raw_ai_words"]
        raw_ai_row = (
            f"| DAIGT AI (`label=1`) | {s['daigt_ai']} | {r['min']} | "
            f"{r['median']:.0f} | {r['mean']:.0f} | {r['max']} |\n"
        )

    return f"""# Dataset

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with: python -m scripts.write_dataset_doc
     Every count below is computed from the CSVs at generation time. -->

Generated {s['generated_at']} from `app/data/{s['daigt_path']}` and
`app/data/{s['polished_path']}`.

## Dataset Summary

| Group | Count | Share |
|---|---:|---:|
| Human-written (DAIGT, `label=0`) | {s['daigt_human']} | {100.0 * s['daigt_human'] / s['total']:.1f}% |
| AI-generated, raw (DAIGT, `label=1`) | {s['daigt_ai']} | {100.0 * s['daigt_ai'] / s['total']:.1f}% |
| AI-polished (derived, `label=1`) | {s['polished']} | {s['pct_polished_of_total']:.1f}% |
| **Total** | **{s['total']}** | **100%** |

Class balance: **{s['pct_human']:.1f}% human ({s['daigt_human']}) vs
{s['pct_ai']:.1f}% AI ({s['ai_total']})**, where the AI side is
{s['daigt_ai']} raw plus {s['polished']} polished.

{balance_note}

Essay lengths, in words:

| Group | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| Human (DAIGT) | {s['daigt_human']} | {hw['min']} | {hw['median']:.0f} | {hw['mean']:.0f} | {hw['max']} |
{raw_ai_row}| AI-polished | {s['polished']} | {pw['min']} | {pw['median']:.0f} | {pw['mean']:.0f} | {pw['max']} |
| ↳ their human originals | {s['polished']} | {ow['min']} | {ow['median']:.0f} | {ow['mean']:.0f} | {ow['max']} |

Everything shorter than 50 words was dropped during normalization: below that,
per-sentence perplexity and burstiness are too noisy to carry signal.

**The polished essays are not an independent sample.** All {s['polished']} were
produced by rewriting human essays drawn from the same DAIGT pool, and
{s['polished_originals_in_human_set']} of their originals are still counted in
the human total above. The same underlying writing therefore appears on both
sides of the label. That is deliberate — it is what isolates *polish* as the
only difference — but it means the polished subset cannot be treated as
independent evidence, and any train/test split must keep an essay and its
polished twin on the same side of the split or the model will simply memorize
the pair.

## Sources

### DAIGT — Kaggle "LLM - Detect AI Generated Text"

{s['daigt_rows']} essays after normalization ({s['daigt_human']} human,
{s['daigt_ai']} AI). The corpus is overwhelmingly **argumentative and
persuasive student writing** responding to a fixed, small set of assigned
prompts — the kind of essay produced for a standardized writing assessment.
{prompt_line}

That shape has direct consequences for what a model trained on it learns:

- **STEM-style writing is underrepresented.** Lab reports, methods sections,
  mathematical exposition, and technical documentation are essentially absent.
  Their register — passive voice, formulaic structure, low lexical variety — is
  exactly the register this detector reads as machine-like, so the false-positive
  risk there is not hypothetical.
- **Creative and narrative writing is underrepresented.** Personal statements,
  short fiction, and reflective essays carry very different burstiness and
  perplexity profiles from persuasive argument.
- **Prompt coverage is narrow.** Essays answering prompts outside the
  competition's original set are out of distribution, and admissions essays —
  the stated target of this project — are *not* the genre this corpus contains.

### AI-polished subset

{s['polished']} essays, built by `scripts/build_polished_set.py`: it samples
{s['polished']} human essays (`random_state=42`, so the sample is reproducible)
and has an LLM lightly rewrite each one, rotating three prompt variants so the
outputs do not all cluster around a single instruction.

| Prompt variant | Essays |
|---|---:|
{variant_rows}

These are the hard cases — human ideas, human structure, human voice, with only
the surface machine-touched.

**This subset is small on purpose, and it is small in absolute terms:**
{s['polished']} essays out of {s['total']} total
(**{s['pct_polished_of_total']:.1f}%** of the dataset, and
{s['pct_polished_of_ai']:.1f}% of the AI-labelled side). Each essay costs an API
call, so this is a probe, not a training corpus. Treat any metric computed on
{s['polished']} examples as indicative and wide-error-barred, not as a
measurement — a handful of essays moving from correct to incorrect swings the
percentage by several points.

## What this dataset does NOT cover

Concrete gaps, not a generic disclaimer. Each of these is a case where a
confident-looking score should not be trusted:

1. **Prompts and topics outside DAIGT's original competition set.** The corpus
   answers a fixed set of assigned prompts. Admissions essays, cover letters,
   research abstracts, and coursework on unrelated subjects are all out of
   distribution — including the admissions-essay use case this project targets.

2. **Non-English and code-switched writing.** The corpus is English-only. GPT-2,
   which computes the perplexity signal, is English-trained. Any essay
   containing sustained non-English passages or intra-sentential code-switching
   will register as high-perplexity for reasons that have nothing to do with
   authorship, and the tool has no way to tell those apart.

3. **Writing by non-native English speakers.** This is a real gap, not a
   caveat. The source data carries **no language-background label at any point**
   — the competition file does not include one, so it cannot be measured,
   controlled for, or reported. This matters more than the other gaps: L2
   writing tends toward simpler syntax, more formulaic phrasing, and lower
   lexical variety, which is precisely the profile this detector scores as
   machine-like. The dataset cannot tell you whether the tool is biased against
   these writers, and the honest position is that it may well be and this
   dataset cannot rule it out.

4. **AI text from models other than those in DAIGT.** The competition's AI
   essays were produced by the generation of models available during the
   competition, and the labelled file does not record which model wrote which
   essay — so per-model performance cannot be broken out even in principle.
   Detectors of this kind learn a generator's statistical fingerprint, and
   fingerprints differ between model families and shift between versions. A
   newer or simply different model's output is not represented here, and
   performance on it is unmeasured. This is the gap most likely to degrade
   silently over time as deployed models change.

5. **Only one depth of AI editing.** The polished subset covers **light polish
   only** — grammar, flow, sentence variety, word choice, with the instruction
   to preserve meaning and voice. Not represented: heavy rewriting,
   paraphrase-level restatement, AI-generated drafts that a human then edited,
   sentence-by-sentence rewriting, or a human-AI back-and-forth. The middle of
   the human↔AI spectrum is mostly unmeasured, and it is where real essays
   actually sit.

6. **Anything about intent.** Nothing in the data distinguishes permitted
   assistance (a grammar checker, a writing centre) from prohibited
   substitution. The tool measures textual resemblance and nothing else.

## Why this still supports the project's claims

These gaps are the reason for two specific design decisions, not an apology
appended after the fact. First, EVALUATION.md reports accuracy **separately for
the DAIGT base and the AI-polished subset** rather than as one blended figure: a
single number would be dominated by the {s['daigt_rows']}-essay base and would
hide performance on the {s['polished']} cases that actually resemble the hard
real-world scenario, while the base's own
{s['imbalance_ratio']:.1f}:1 class skew means an unqualified accuracy figure
overstates capability regardless. Second, the API and UI never assert authorship:
`/analyze` returns `overall_probability` alongside a `confidence_label` sentence
and a `label` of "Likely AI" or "Likely Human", every sentence carries a
resemblance `phrase` rather than a bare score, and the words "definitely" and
"detected" appear nowhere in the response. That framing is what this dataset
supports. A corpus of {s['total']} essays — narrow in genre, single-language,
unlabelled for writer background, covering one generation of AI models and one
depth of AI editing — can justify a calibrated statement about resemblance to
its own examples. It cannot justify a verdict about a specific student, and the
product does not make one.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daigt", type=Path, default=DEFAULT_DAIGT)
    parser.add_argument("--polished", type=Path, default=DEFAULT_POLISHED)
    parser.add_argument("--raw-train", type=Path, default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    stats = collect(args.daigt, args.polished, args.raw_train)

    logger.info("Computed from the real CSVs:")
    logger.info("  human (DAIGT)      : %d", stats["daigt_human"])
    logger.info("  AI raw (DAIGT)     : %d", stats["daigt_ai"])
    logger.info("  AI polished        : %d", stats["polished"])
    logger.info("  total              : %d", stats["total"])
    logger.info(
        "  balance            : %.1f%% human / %.1f%% AI (%.1f:1 toward %s)",
        stats["pct_human"],
        stats["pct_ai"],
        stats["imbalance_ratio"],
        stats["majority_class"],
    )

    args.out.write_text(render(stats), encoding="utf-8")
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()

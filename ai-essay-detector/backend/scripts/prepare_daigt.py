"""Normalize the raw DAIGT competition data into a single labelled CSV.

    python -m scripts.prepare_daigt                          # every CSV in app/data/raw
    python -m scripts.prepare_daigt --raw-dir path/to/unzip  # a different unzip location
    python -m scripts.prepare_daigt --data train_essays.csv  # one specific file

Get the raw data first:

    pip install kaggle
    kaggle competitions download -c llm-detect-ai-generated-text
    unzip llm-detect-ai-generated-text.zip -d app/data/raw

The competition ships three CSVs and they do NOT share a schema:

    train_essays.csv   id, prompt_id, text, generated   <- the only labelled one
    train_prompts.csv  prompt_id, prompt_name, instructions, source_text
    test_essays.csv    id, prompt_id, text              <- unlabelled

So this script does not assume a layout. It prints the real columns of every
file it opens, resolves the text and label columns against an explicit
candidate list, and skips any file that has no label column rather than
guessing one. Anything unresolved is a hard error listing what was actually
found, so a schema change surfaces immediately instead of silently producing a
mislabelled dataset.

Output: app/data/daigt_essays.csv with exactly two columns -- text, label
(0 = human, 1 = AI).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/prepare_daigt.py` as well as `python -m scripts.prepare_daigt`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("prepare_daigt")

DATA_DIR = BACKEND_ROOT / "app" / "data"
DEFAULT_RAW_DIR = DATA_DIR / "raw"
DEFAULT_OUT = DATA_DIR / "daigt_essays.csv"

# Checked in order; first hit wins. The competition's own names lead.
TEXT_COLUMN_CANDIDATES = ("text", "essay", "full_text", "essay_text", "content")
LABEL_COLUMN_CANDIDATES = (
    "generated",  # what train_essays.csv actually uses
    "label",
    "is_generated",
    "ai_generated",
    "target",
    "class",
)

# Whitespace-delimited tokens. Deliberately not signals.tokenize_words: that
# counts only alphabetic words and would drag torch into a pure-CSV script.
_WORD_RE = re.compile(r"\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def resolve_column(
    columns: list[str], candidates: tuple[str, ...], override: str | None
) -> str | None:
    """Map a real column name to a role, case-insensitively.

    Returns None when nothing matches, so the caller can decide whether that is
    a skip (no label column) or a hard error (no text column).
    """
    lookup = {c.lower().strip(): c for c in columns}

    if override:
        if override in columns:
            return override
        if override.lower() in lookup:
            return lookup[override.lower()]
        raise SystemExit(
            f"Column {override!r} not found. Available columns: {columns}"
        )

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def load_one(path: Path, text_override: str | None, label_override: str | None):
    """Read one raw CSV and return a (text, label) frame, or None to skip it."""
    frame = pd.read_csv(path)
    columns = list(frame.columns)
    logger.info("%s: %d rows, columns = %s", path.name, len(frame), columns)

    text_col = resolve_column(columns, TEXT_COLUMN_CANDIDATES, text_override)
    label_col = resolve_column(columns, LABEL_COLUMN_CANDIDATES, label_override)

    if text_col is None and label_col is None:
        # train_prompts.csv lands here: it holds prompt metadata, not essays.
        # Not an error, just not our file.
        logger.warning(
            "%s: neither a text nor a label column -- skipping (not an essay "
            "file). Pass --text-col/--label-col if this is wrong.",
            path.name,
        )
        return None

    if text_col is None:
        # A label but no text means the file *is* essay-shaped and we failed to
        # map it -- that is a genuine error, not something to skip past.
        raise SystemExit(
            f"{path.name}: found label column {label_col!r} but no text column. "
            f"Looked for {list(TEXT_COLUMN_CANDIDATES)}, found {columns}. "
            "Pass --text-col to map it explicitly."
        )

    if label_col is None:
        # test_essays.csv lands here -- real essays, no labels. Skipping is
        # correct; inventing a label would poison the training set.
        logger.warning(
            "%s: has text but no label column (looked for %s) -- skipping. "
            "Pass --label-col if this file really is labelled.",
            path.name,
            list(LABEL_COLUMN_CANDIDATES),
        )
        return None

    logger.info("%s: mapped text=%r label=%r", path.name, text_col, label_col)
    out = frame[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out["_source"] = path.name
    return out


def coerce_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Force labels to integer 0/1, dropping rows that are neither.

    Handles ints, floats, "0"/"1" strings, and true/false spellings; anything
    else is reported and dropped rather than silently cast.
    """
    truthy = {"1", "true", "yes", "ai", "generated"}
    falsy = {"0", "false", "no", "human"}

    def to_binary(value) -> int | None:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value) if float(value) in (0.0, 1.0) else None
        token = str(value).strip().lower()
        if token in truthy:
            return 1
        if token in falsy:
            return 0
        return None

    coerced = frame["label"].map(to_binary)
    unusable = coerced.isna()
    if unusable.any():
        bad = frame.loc[unusable, "label"].astype(str).value_counts().head(5)
        logger.warning(
            "Dropping %d row(s) with an unrecognised label. Most common: %s",
            int(unusable.sum()),
            dict(bad),
        )

    frame = frame.loc[~unusable].copy()
    frame["label"] = coerced.loc[~unusable].astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory holding the unzipped competition CSVs.",
    )
    parser.add_argument(
        "--data", nargs="*", type=Path, help="Specific CSV file(s) instead of --raw-dir."
    )
    parser.add_argument("--text-col", default=None, help="Override the text column.")
    parser.add_argument("--label-col", default=None, help="Override the label column.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Drop essays shorter than this. Below ~50 words the sentence-level "
        "signals (burstiness, per-sentence perplexity) are too noisy to trust.",
    )
    args = parser.parse_args()

    if args.data:
        paths = [Path(p) for p in args.data]
    else:
        paths = sorted(args.raw_dir.glob("*.csv"))
        if not paths:
            raise SystemExit(
                f"No CSVs in {args.raw_dir}. Download and unzip the competition "
                "data first:\n"
                "  kaggle competitions download -c llm-detect-ai-generated-text\n"
                f"  unzip llm-detect-ai-generated-text.zip -d {args.raw_dir}"
            )

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"File(s) not found: {[str(p) for p in missing]}")

    logger.info("Inspecting %d file(s) ...", len(paths))
    frames = [
        f
        for f in (load_one(p, args.text_col, args.label_col) for p in paths)
        if f is not None
    ]
    if not frames:
        raise SystemExit(
            "No labelled CSV found. train_essays.csv is the labelled file in this "
            "competition; test_essays.csv and train_prompts.csv are not."
        )

    data = pd.concat(frames, ignore_index=True)
    total_in = len(data)
    logger.info("Combined: %d rows from %d labelled file(s)", total_in, len(frames))

    # --- Clean -----------------------------------------------------------
    data = coerce_labels(data)
    after_labels = len(data)

    # fillna before astype: on pandas 3 astype(str) keeps NaN as a missing
    # value rather than the literal "nan", and a null essay is an empty one.
    data["text"] = data["text"].fillna("").astype(str)
    # Collapse internal whitespace for the dedup key only; the stored text keeps
    # its original spacing so downstream sentence spans stay meaningful.
    normalized = data["text"].str.strip().map(lambda t: _WHITESPACE_RE.sub(" ", t))

    empty = normalized.str.len() == 0
    data, normalized = data.loc[~empty], normalized.loc[~empty]
    after_empty = len(data)

    word_counts = normalized.map(lambda t: len(_WORD_RE.findall(t)))
    too_short = word_counts < args.min_words
    data, normalized = data.loc[~too_short], normalized.loc[~too_short]
    after_short = len(data)

    duplicated = normalized.duplicated(keep="first")
    data = data.loc[~duplicated]
    after_dupes = len(data)

    data["text"] = data["text"].str.strip()
    data = data[["text", "label"]].reset_index(drop=True)

    logger.info(
        "Cleaning: %d in -> %d after labels -> %d after empty -> %d after "
        "<%d words -> %d after dedupe",
        total_in,
        after_labels,
        after_empty,
        after_short,
        args.min_words,
        after_dupes,
    )

    if data.empty:
        raise SystemExit("Every row was filtered out; nothing to write.")

    # --- Class balance ---------------------------------------------------
    human = int((data["label"] == 0).sum())
    ai = int((data["label"] == 1).sum())
    total = human + ai

    logger.info("=" * 62)
    logger.info("FINAL CLASS BALANCE")
    logger.info("  human (0): %6d  (%5.2f%%)", human, 100.0 * human / total)
    logger.info("  AI    (1): %6d  (%5.2f%%)", ai, 100.0 * ai / total)
    logger.info("  total    : %6d", total)

    if human == 0 or ai == 0:
        logger.error(
            "  Only one class present -- this cannot train a classifier. "
            "Check that the labelled file was picked up."
        )
    else:
        larger, smaller = max(human, ai), min(human, ai)
        ratio = larger / smaller
        minority = "AI" if ai < human else "human"
        logger.info("  imbalance: %.1f:1 (minority class = %s)", ratio, minority)
        if ratio >= 3.0:
            logger.warning(
                "  HEAVILY IMBALANCED. Record this in DATASET.md: raw accuracy is "
                "misleading here -- a model predicting the majority class every "
                "time scores %.1f%%. Judge it on per-class precision/recall and "
                "ROC-AUC instead, and keep class_weight='balanced' when training.",
                100.0 * larger / total,
            )
    logger.info("=" * 62)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    logger.info("Wrote %s (%d rows, columns = %s)", args.out, len(data), list(data.columns))


if __name__ == "__main__":
    main()

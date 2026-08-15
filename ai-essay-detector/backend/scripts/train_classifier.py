"""Train the essay classifier and write the .joblib the API loads.

    python -m scripts.train_classifier                      # all CSVs in app/data
    python -m scripts.train_classifier --data mydata.csv    # a specific dataset
    python -m scripts.train_classifier --refresh-cache      # force re-extraction

Input CSVs need a text column and a binary label column (default ``text`` and
``label``; 1 = AI-generated, 0 = human). Every essay is scored through the same
``services.signals`` code the API uses, so the feature vector at training time
and the feature vector at inference time cannot diverge.

**Feature extraction is the expensive part** -- it runs GPT-2 over every essay.
The extracted matrix is cached to ``app/data/features_cache.csv``; later runs
load it and go straight to training, so tuning the classifier costs seconds
instead of another full GPT-2 pass. The cache is keyed to the current
``FEATURE_NAMES``; if those change it is rejected rather than silently reused.

**Logistic regression on purpose, not for lack of ambition.** A gradient-boosted
ensemble would likely score a point or two higher and would make the per-feature
explanations in the API a post-hoc reconstruction rather than the model's actual
reasoning. Here the standardised coefficients *are* the explanation: the weights
printed at the end of training are the same numbers `services.classifier` uses
to attribute each essay's score to individual signals. That is what keeps the
explainability honest.

The artifact is a dict, not a bare estimator: it carries the feature order, the
fitted weights, and the sentence-level calibration stats that
``services.classifier`` needs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Allow `python scripts/train_classifier.py` as well as `python -m scripts.train_classifier`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT, get_settings  # noqa: E402
from app.services import model_loader  # noqa: E402
from app.services.signals import FEATURE_NAMES, extract_signals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("train")
for _noisy in ("httpx", "huggingface_hub", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DATA_DIR = BACKEND_ROOT / "app" / "data"
DEFAULT_CACHE = DATA_DIR / "features_cache.csv"

# Bookkeeping columns stored alongside the features. Underscore-prefixed so
# they can never collide with a feature name. They hold the sufficient
# statistics for the pooled sentence-logprob calibration, so the calibration
# can be recomputed exactly from the cache without re-running GPT-2.
META_COLUMNS = ("_sent_count", "_sent_logprob_sum", "_sent_logprob_sumsq")

# A bounded excerpt of each essay, cached alongside its features so
# scripts/generate_evaluation.py can quote misclassified essays without
# re-reading and re-aligning the source CSVs (whose row indices shift whenever
# the dataset is rebuilt). Bounded rather than the full text to keep the cache
# from becoming a second copy of the corpus.
EXCERPT_COLUMN = "_text_excerpt"
EXCERPT_CHARS = 400

FALLBACK_LOGPROB_MEAN = -3.6
FALLBACK_LOGPROB_STD = 1.1


def load_dataset(
    paths: list[Path], text_col: str, label_col: str
) -> pd.DataFrame:
    """Load and concatenate the input CSVs, tagging each row with its source.

    ``source`` lets evaluation split results by origin later -- overall accuracy
    hides the fact that the AI-polished rows are the hard ones. Files that
    already carry a ``source`` column (ai_polished_essays.csv) keep their value;
    anything else is tagged from its filename stem, so daigt_essays.csv becomes
    ``daigt``.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {text_col, label_col} - set(frame.columns)
        if missing:
            raise SystemExit(
                f"{path} is missing column(s) {sorted(missing)}; "
                f"found {list(frame.columns)}"
            )

        if "source" in frame.columns:
            source = frame["source"].astype(str)
            kept = source.value_counts().to_dict()
            logger.info(
                "Loaded %d rows from %s (existing source column: %s)",
                len(frame),
                path.name,
                kept,
            )
        else:
            # daigt_essays.csv has no source column -- add one from the stem.
            stem = path.stem.replace("_essays", "")
            source = pd.Series([stem] * len(frame), index=frame.index)
            logger.info(
                "Loaded %d rows from %s (tagged source=%r)", len(frame), path.name, stem
            )

        subset = frame[[text_col, label_col]].copy()
        subset.columns = ["text", "label"]
        subset["source"] = source.values
        frames.append(subset)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0]
    data["label"] = data["label"].astype(int)
    return data.reset_index(drop=True)


def build_features(data: pd.DataFrame, bundle) -> tuple[pd.DataFrame, Counter]:
    """Score every essay through signals.extract_signals.

    One bad row must not cost the whole run -- extraction is wrapped per essay,
    failures are counted by reason and reported at the end. Returns the feature
    frame (features + label + source + calibration bookkeeping) and the failure
    tally.
    """
    rows: list[dict] = []
    failures: Counter = Counter()

    iterator = tqdm(
        data.itertuples(index=True),
        total=len(data),
        desc="Extracting features",
        unit="essay",
        dynamic_ncols=True,
    )

    for row in iterator:
        text = row.text
        if not text or not text.strip():
            failures["empty text"] += 1
            continue

        try:
            signals = extract_signals(text, bundle)
        except Exception as exc:  # noqa: BLE001 - one bad essay, not a crash
            failures[f"extraction error: {type(exc).__name__}"] += 1
            logger.debug("Row %s failed: %s", row.Index, exc, exc_info=True)
            continue

        if not signals.sentence_scores:
            # Nothing GPT-2 could score: no sentences survived segmentation.
            failures["no scoreable sentences"] += 1
            continue

        if signals.word_count == 0:
            # Punctuation or symbols only -- every stylistic feature is 0 and
            # the row would be pure noise in training.
            failures["no words after tokenization"] += 1
            continue

        vector = signals.vector()
        if not np.all(np.isfinite(vector)):
            failures["non-finite feature value"] += 1
            continue

        record = dict(zip(FEATURE_NAMES, (float(v) for v in vector)))
        record["label"] = int(row.label)
        record["source"] = str(row.source)

        # Sufficient statistics for the pooled human sentence-logprob
        # calibration. Stored per row so the pooled mean/std can be
        # reconstructed exactly from the cache, without re-running GPT-2.
        scored = [s.mean_logprob for s in signals.sentence_scores if s.token_count > 0]
        record["_sent_count"] = len(scored)
        record["_sent_logprob_sum"] = float(sum(scored))
        record["_sent_logprob_sumsq"] = float(sum(v * v for v in scored))
        record[EXCERPT_COLUMN] = text[:EXCERPT_CHARS]

        rows.append(record)

    if not rows:
        raise SystemExit("No essays survived feature extraction.")

    columns = (
        list(FEATURE_NAMES) + ["label", "source"] + list(META_COLUMNS) + [EXCERPT_COLUMN]
    )
    return pd.DataFrame(rows, columns=columns), failures


def calibration_from_features(features: pd.DataFrame) -> tuple[float, float]:
    """Pooled mean/std of human sentence log-probabilities.

    Computed from the per-row sufficient statistics so the figure is identical
    whether features were just extracted or loaded from cache.
    """
    human = features[features["label"] == 0]
    n = float(human["_sent_count"].sum())
    if n < 2:
        logger.warning("Too few human sentences to calibrate; using defaults.")
        return FALLBACK_LOGPROB_MEAN, FALLBACK_LOGPROB_STD

    total = float(human["_sent_logprob_sum"].sum())
    total_sq = float(human["_sent_logprob_sumsq"].sum())
    mean = total / n
    variance = max(total_sq / n - mean * mean, 0.0)
    # Bessel correction, to match np.std(ddof=1) on the pooled sentences.
    variance *= n / (n - 1.0)
    return mean, max(variance**0.5, 1e-6)


def load_cache(path: Path) -> pd.DataFrame | None:
    """Load the cached feature matrix, or None if it is absent or stale."""
    if not path.exists():
        return None

    cached = pd.read_csv(path)
    required = set(FEATURE_NAMES) | {"label", "source"} | set(META_COLUMNS)
    missing = required - set(cached.columns)
    if not missing and EXCERPT_COLUMN not in cached.columns:
        # Predates the excerpt column. Fine for training; only
        # generate_evaluation.py needs it, and it says so if absent.
        logger.warning(
            "Cache has no %s column; re-run with --refresh-cache before "
            "generating the evaluation report with excerpts.",
            EXCERPT_COLUMN,
        )
    if missing:
        raise SystemExit(
            f"{path} is stale: missing column(s) {sorted(missing)}.\n"
            "signals.FEATURE_NAMES has changed since it was written. Re-extract "
            "with:\n  python -m scripts.train_classifier --refresh-cache"
        )

    logger.info("Loaded %d cached feature rows from %s", len(cached), path.name)
    logger.info("Skipping GPT-2 extraction. Use --refresh-cache to force a re-run.")
    return cached


def format_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Confusion matrix with the axes spelled out.

    A bare 2x2 array is easy to read backwards. The two error cells matter in
    opposite ways here: a false positive accuses a student who wrote their own
    essay, a false negative just misses one. They are labelled.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = matrix
    return (
        "\n                        predicted\n"
        "                     human      AI\n"
        f"    actual human  {tn:>7} {fp:>7}   <- {fp} false positive(s): human essay called AI\n"
        f"           AI     {fn:>7} {tp:>7}   <- {fn} false negative(s): AI essay called human"
    )


def report_feature_weights(pipeline: Pipeline, feature_names: tuple[str, ...]) -> dict:
    """Print the fitted coefficients, largest influence first.

    This is the reason the model is a logistic regression and not something
    opaque: the coefficients are the explanation. Because the features were
    standardised first, the magnitudes are directly comparable -- a coefficient
    of 1.2 really does move the logit twice as hard as one of 0.6, which is what
    lets per-feature contributions be reported at inference time without
    inventing a post-hoc story.
    """
    final = pipeline.named_steps["clf"]
    coefficients = np.ravel(final.coef_)
    weights = {name: float(c) for name, c in zip(feature_names, coefficients)}

    logger.info("=" * 62)
    logger.info("FEATURE WEIGHTS (standardised logistic-regression coefficients)")
    logger.info("  positive -> pushes toward AI     negative -> pushes toward human")
    logger.info("-" * 62)
    for name, weight in sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True):
        logger.info(
            "  %+8.4f  %-26s %s",
            weight,
            name,
            "AI" if weight > 0 else "human" if weight < 0 else "-",
        )
    logger.info("  intercept: %+.4f", float(np.ravel(final.intercept_)[0]))
    logger.info("=" * 62)
    return weights


def report_cross_validation(
    pipeline_factory, X: np.ndarray, y: np.ndarray, folds: int, seed: int
) -> None:
    """Report metrics from stratified k-fold CV rather than one lucky split.

    A single 80/20 split's accuracy can move several points depending on which
    essays happen to land in the test fold. The mean and spread across folds
    say how much of a reported number is signal.
    """
    smallest_class = int(np.bincount(y).min())
    if smallest_class < folds:
        logger.warning(
            "Smallest class has %d rows -- fewer than %d folds. Reducing to %d.",
            smallest_class,
            folds,
            max(smallest_class, 2),
        )
        folds = max(smallest_class, 2)
    if folds < 2:
        logger.error("Not enough data per class for cross-validation; skipping.")
        return

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    accuracies = cross_val_score(pipeline_factory(), X, y, cv=cv, scoring="accuracy")
    logger.info("=" * 62)
    logger.info("%d-FOLD CROSS-VALIDATED PERFORMANCE (the honest numbers)", folds)
    logger.info(
        "  accuracy : %.4f +/- %.4f  (folds: %s)",
        accuracies.mean(),
        accuracies.std(),
        ", ".join(f"{a:.3f}" for a in accuracies),
    )

    try:
        aucs = cross_val_score(pipeline_factory(), X, y, cv=cv, scoring="roc_auc")
        logger.info("  ROC-AUC  : %.4f +/- %.4f", aucs.mean(), aucs.std())
    except ValueError as exc:
        logger.warning("  ROC-AUC unavailable: %s", exc)

    predictions = cross_val_predict(pipeline_factory(), X, y, cv=cv)
    logger.info(
        "  pooled out-of-fold report:\n%s",
        classification_report(y, predictions, zero_division=0),
    )
    # Every row appears here exactly once, as an out-of-fold prediction -- so
    # this counts real errors on unseen data, not the single-split version.
    logger.info("  pooled out-of-fold confusion matrix:%s", format_confusion(y, predictions))
    logger.info("=" * 62)


def report_by_source(
    pipeline_factory,
    X: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    folds: int,
    seed: int,
) -> None:
    """Out-of-fold accuracy broken out per source.

    Blended accuracy is dominated by whichever source has the most rows; the
    AI-polished rows are the hard cases and deserve their own line.
    """
    smallest_class = int(np.bincount(y).min())
    folds = min(folds, max(smallest_class, 2))
    if folds < 2:
        return

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = cross_val_predict(pipeline_factory(), X, y, cv=cv)

    logger.info("Out-of-fold accuracy by source:")
    for source in sorted(set(sources)):
        mask = sources == source
        correct = int((predictions[mask] == y[mask]).sum())
        total = int(mask.sum())
        logger.info(
            "  %-14s %4d/%-4d = %.4f   (labels present: %s)",
            source,
            correct,
            total,
            correct / total if total else 0.0,
            sorted(set(y[mask].tolist())),
        )


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="*", type=Path)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--out", type=Path, default=settings.resolved_model_path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore any cached features and re-run GPT-2 extraction.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Features: from cache when possible ------------------------------
    features = None if args.refresh_cache else load_cache(args.cache)
    if args.refresh_cache and args.cache.exists():
        logger.info("--refresh-cache: ignoring %s", args.cache.name)

    if features is None:
        paths = args.data or sorted(
            p
            for p in DATA_DIR.glob("*.csv")
            # Never feed the cache back in as training input.
            if p.name != args.cache.name
        )
        if not paths:
            raise SystemExit(f"No CSVs found in {DATA_DIR}. Add a dataset or pass --data.")

        data = load_dataset([Path(p) for p in paths], args.text_col, args.label_col)
        logger.info(
            "Dataset: %d essays (%d AI / %d human) across sources %s",
            len(data),
            int((data["label"] == 1).sum()),
            int((data["label"] == 0).sum()),
            data["source"].value_counts().to_dict(),
        )
        if data["label"].nunique() < 2:
            raise SystemExit("Dataset needs both classes (0 and 1) to train.")

        logger.info("Loading GPT-2 for feature extraction ...")
        bundle = model_loader.load_gpt2(settings)

        features, failures = build_features(data, bundle)

        # --- Extraction failure summary ----------------------------------
        dropped = sum(failures.values())
        if dropped:
            logger.warning("-" * 62)
            logger.warning(
                "Dropped %d of %d rows during extraction (%.1f%%):",
                dropped,
                len(data),
                100.0 * dropped / len(data),
            )
            for reason, count in failures.most_common():
                logger.warning("  %5d  %s", count, reason)
            logger.warning("-" * 62)
        else:
            logger.info("All %d rows extracted cleanly.", len(data))

        args.cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.cache, index=False)
        logger.info(
            "Cached %d feature rows to %s -- later runs skip GPT-2 entirely.",
            len(features),
            args.cache,
        )

    # --- Assemble matrices ------------------------------------------------
    X = features[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)
    sources = features["source"].astype(str).to_numpy()
    logprob_mean, logprob_std = calibration_from_features(features)

    if len(np.unique(y)) < 2:
        raise SystemExit("Need both classes present after extraction to train.")

    # --- Class balance, printed before training --------------------------
    n_human = int((y == 0).sum())
    n_ai = int((y == 1).sum())
    total = n_human + n_ai
    logger.info("=" * 62)
    logger.info("TRAINING DATA CLASS BALANCE")
    logger.info("  label=0 (human): %5d  (%5.2f%%)", n_human, 100.0 * n_human / total)
    logger.info("  label=1 (AI)   : %5d  (%5.2f%%)", n_ai, 100.0 * n_ai / total)
    logger.info("  total          : %5d", total)
    larger = max(n_human, n_ai)
    ratio = larger / max(min(n_human, n_ai), 1)
    if ratio >= 1.5:
        logger.warning(
            "  Imbalanced %.1f:1. Always predicting the majority class scores "
            "%.1f%% -- read the accuracy below against that floor, not against "
            "50%%. class_weight='balanced' is set.",
            ratio,
            100.0 * larger / total,
        )
    logger.info("  by source: %s", pd.Series(sources).value_counts().to_dict())
    logger.info("=" * 62)

    def make_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    # --- Reported metrics: cross-validated, not one lucky split ----------
    report_cross_validation(make_pipeline, X, y, args.folds, args.seed)
    report_by_source(make_pipeline, X, y, sources, args.folds, args.seed)

    # --- Final model: fit on the train split -----------------------------
    stratify = y if np.bincount(y).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Final model fit on %d rows (%d held out).", len(y_train), len(y_test))

    if len(np.unique(y_test)) > 1:
        test_predictions = pipeline.predict(X_test)
        held_out = pipeline.score(X_test, y_test)
        auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        logger.info(
            "Held-out split (secondary -- one split, wide error bars): "
            "accuracy %.4f, ROC-AUC %.4f",
            held_out,
            auc,
        )
        logger.info(
            "  held-out report:\n%s",
            classification_report(y_test, test_predictions, zero_division=0),
        )
        logger.info(
            "  held-out confusion matrix:%s",
            format_confusion(y_test, test_predictions),
        )

    # The coefficients of the model actually being saved.
    feature_weights = report_feature_weights(pipeline, FEATURE_NAMES)

    # --- Save -------------------------------------------------------------
    payload = {
        # 'model' is the full pipeline: scaler + estimator, so predict_proba
        # works straight off it. 'scaler' is the same fitted object, exposed
        # separately for inspection and feature-contribution maths.
        "model": pipeline,
        "scaler": pipeline.named_steps["scaler"],
        "feature_names": list(FEATURE_NAMES),
        # Same numbers as model.coef_, stored plainly so Phase 5 explainability
        # can read a name -> weight mapping without unwrapping the pipeline.
        "feature_weights": feature_weights,
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "sentence_logprob_mean": logprob_mean,
        "sentence_logprob_std": logprob_std,
        "n_training_essays": int(len(y_train)),
        "class_balance": {"human": n_human, "ai": n_ai},
        "sources": pd.Series(sources).value_counts().to_dict(),
        "gpt2_model_name": settings.gpt2_model_name,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    logger.info("Wrote %s (version %s)", out, payload["version"])
    logger.info(
        "Artifact carries feature_names (%d), so classifier.py builds its "
        "vector from this file rather than a second hardcoded list.",
        len(FEATURE_NAMES),
    )


if __name__ == "__main__":
    main()

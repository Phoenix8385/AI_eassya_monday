"""Trained-classifier loading and essay scoring.

Loads the ``.joblib`` artifact once at startup (same singleton discipline as
``model_loader``), then scores essays from the feature vector that
``services.signals`` produces and turns the result into per-sentence evidence.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import Settings, get_settings
from app.models.schemas import (
    AnalyzeResponse,
    FeatureContribution,
    SentenceInsight,
    SignalSummary,
)
from app.services import signals as signals_mod
from app.services.model_loader import get_gpt2
from app.services.signals import FEATURE_NAMES, SignalBundle

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_bundle: "ClassifierBundle | None" = None

# Fallback reference distribution for per-sentence log-probability, used only
# when the artifact was saved without calibration stats. Derived from GPT-2
# scoring of ordinary English prose.
_FALLBACK_SENTENCE_LOGPROB_MEAN = -3.6
_FALLBACK_SENTENCE_LOGPROB_STD = 1.1

# The point where the wording flips between the AI-leaning and human-leaning
# phrasings. This is a presentation pivot, not a verdict boundary: both sides
# describe the same probability, so a score sitting near it produces a
# near-50% sentence either way rather than a confident-sounding call.
_LABEL_PIVOT = 0.5


@dataclass(frozen=True)
class ClassifierBundle:
    """The trained estimator plus the metadata saved alongside it."""

    estimator: Any
    feature_names: tuple[str, ...]
    version: str
    sentence_logprob_mean: float
    sentence_logprob_std: float
    source_path: Path


def load_classifier(settings: Settings | None = None) -> ClassifierBundle:
    """Load the trained model once and cache it."""
    global _bundle

    if _bundle is not None:
        return _bundle

    with _lock:
        if _bundle is not None:
            return _bundle

        settings = settings or get_settings()
        path = settings.resolved_model_path

        if not path.exists():
            raise FileNotFoundError(
                f"No trained classifier at {path}. Generate it with "
                "`python -m scripts.train` or point MODEL_PATH at an existing artifact."
            )

        started = time.perf_counter()
        payload = joblib.load(path)

        # Accept either a bare estimator or the dict the training script writes.
        if isinstance(payload, dict):
            estimator = payload.get("estimator") or payload.get("model")
            feature_names = tuple(payload.get("feature_names") or FEATURE_NAMES)
            version = str(payload.get("version", "unknown"))
            logprob_mean = float(
                payload.get("sentence_logprob_mean", _FALLBACK_SENTENCE_LOGPROB_MEAN)
            )
            logprob_std = float(
                payload.get("sentence_logprob_std", _FALLBACK_SENTENCE_LOGPROB_STD)
            )
        else:
            estimator = payload
            feature_names = FEATURE_NAMES
            version = "unknown"
            logprob_mean = _FALLBACK_SENTENCE_LOGPROB_MEAN
            logprob_std = _FALLBACK_SENTENCE_LOGPROB_STD

        if estimator is None or not hasattr(estimator, "predict"):
            raise ValueError(f"Artifact at {path} does not contain a usable estimator.")

        # Scoring reads predict_proba(...)[0][1] -- a probability, never a 0/1
        # label. An estimator without predict_proba cannot support that, so it
        # is rejected here rather than silently degrading to predict() at
        # request time and throwing the confidence information away.
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                f"Estimator at {path} has no predict_proba(); this API reports "
                "probabilities, not bare labels. Retrain with a probabilistic "
                "estimator (e.g. LogisticRegression)."
            )

        # Column 1 of predict_proba is only P(AI) when the classes are ordered
        # [0, 1]. sklearn sorts classes_, so this holds for 0/1 labels -- assert
        # it at load so a differently-labelled artifact cannot invert every
        # score in a way nothing downstream would notice.
        raw_classes = list(getattr(estimator, "classes_", [0, 1]))
        try:
            classes = [int(c) for c in raw_classes]
        except (TypeError, ValueError):
            classes = None
        if classes != [0, 1]:
            raise ValueError(
                f"Estimator at {path} has classes_={raw_classes}; expected "
                "[0, 1] (0 = human, 1 = AI) so that predict_proba column 1 is "
                "P(AI). Retrain with integer 0/1 labels."
            )

        # The artifact's feature list is authoritative -- scoring builds its
        # vector in this order, by name. That removes the second source of
        # truth: there is no separate hardcoded order here to drift from the
        # training script. What still has to hold is that signals.py can
        # actually produce every feature the model was fitted on.
        unknown = [name for name in feature_names if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"Artifact at {path} was trained on feature(s) {unknown} that "
                "signals.py no longer produces. Retrain against the current code."
            )

        if tuple(feature_names) != FEATURE_NAMES:
            # Satisfiable but worth saying out loud: a reordered or reduced
            # feature set scores correctly, it just means the artifact predates
            # a signals.py change.
            logger.warning(
                "Artifact feature list differs from signals.FEATURE_NAMES "
                "(%d vs %d features). Scoring follows the artifact's order; "
                "retrain to pick up the newer features.",
                len(feature_names),
                len(FEATURE_NAMES),
            )

        _bundle = ClassifierBundle(
            estimator=estimator,
            feature_names=tuple(feature_names),
            version=version,
            sentence_logprob_mean=logprob_mean,
            sentence_logprob_std=max(logprob_std, 1e-6),
            source_path=path,
        )
        logger.info(
            "Classifier '%s' loaded from %s in %.2fs",
            version,
            path,
            time.perf_counter() - started,
        )
        return _bundle


def get_classifier() -> ClassifierBundle:
    if _bundle is None:
        raise RuntimeError(
            "Classifier is not loaded. It is loaded once during application "
            "startup; check the boot logs for the failure."
        )
    return _bundle


def is_loaded() -> bool:
    return _bundle is not None


def reset() -> None:
    """Drop the singleton. Intended for tests only."""
    global _bundle
    with _lock:
        _bundle = None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def confidence_label(probability: float) -> str:
    """Turn a raw probability into a sentence a reader can act on.

    The output is always a full sentence framed as resemblance to training
    examples -- never a verdict, never a bare percentage standing alone. The
    two branches deliberately describe the *same* number from opposite ends so
    a low AI probability reads as positive evidence of human patterns rather
    than as a weak accusation.
    """
    pct = round(probability * 100)
    if probability >= _LABEL_PIVOT:
        return (
            f"This essay shares {pct}% of its measured patterns with "
            "AI-generated examples in the training data."
        )
    return (
        f"This essay's patterns align {100 - pct}% with human-written "
        "examples in the training data."
    )


def _short_label(probability: float) -> str:
    """Short tag shown beside the probability, never on its own."""
    return "Likely AI" if probability >= _LABEL_PIVOT else "Likely Human"


def _sentence_phrase(likelihood: float, settings: Settings) -> str:
    """A resemblance phrase for one sentence.

    Every sentence gets one, so no per-sentence score is ever presented as a
    bare number with nothing to interpret it.
    """
    if likelihood >= settings.sentence_flag_threshold:
        return "closely resembles AI-pattern examples"
    if likelihood >= _LABEL_PIVOT:
        return "leans toward AI-pattern examples"
    return "resembles human-written examples"


def _final_estimator(estimator: Any) -> Any:
    """Unwrap a Pipeline to the estimator that carries the coefficients."""
    steps = getattr(estimator, "steps", None)
    return steps[-1][1] if steps else estimator


def _feature_contributions(
    estimator: Any,
    bundle: SignalBundle,
    feature_names: tuple[str, ...],
    limit: int = 5,
) -> list[FeatureContribution]:
    """Signed per-feature push toward 'AI', when the model exposes weights.

    For a linear model this is the standardised coefficient times the
    standardised value, which is the term's actual share of the logit. For a
    tree ensemble only unsigned importances exist, so the sign is dropped.

    ``feature_names`` comes from the artifact, so the coefficients line up with
    the same vector the model was scored on.
    """
    final = _final_estimator(estimator)
    values = [float(bundle.features[name]) for name in feature_names]

    # Reuse the training scaler's statistics so contributions are comparable.
    scaler = None
    for _, step in getattr(estimator, "steps", []) or []:
        if hasattr(step, "mean_") and hasattr(step, "scale_"):
            scaler = step
            break

    if hasattr(final, "coef_"):
        coefficients = np.ravel(final.coef_)
        raw = np.asarray(values, dtype=float)
        if scaler is not None:
            standardized = (raw - scaler.mean_) / scaler.scale_
        else:
            standardized = raw
        weighted = coefficients * standardized
    elif hasattr(final, "feature_importances_"):
        weighted = np.ravel(final.feature_importances_)
    else:
        return []

    if len(weighted) != len(feature_names):
        return []

    ranked = sorted(
        range(len(feature_names)), key=lambda i: abs(weighted[i]), reverse=True
    )
    return [
        FeatureContribution(
            name=feature_names[i],
            value=round(float(values[i]), 6),
            contribution=round(float(weighted[i]), 6),
        )
        for i in ranked[:limit]
    ]


def _sentence_ai_likelihood(
    mean_logprob: float, bundle: ClassifierBundle, settings: Settings
) -> float:
    """Map a sentence's mean log-probability to a 0-1 AI likelihood.

    A sentence GPT-2 finds unusually *easy* to predict (high log-prob relative
    to human writing) is the AI-like case, so the z-score rises with
    predictability. The offset matters: the reference distribution is built
    from human sentences, so without it half of a genuinely human essay sits
    above the mean by construction and would be flagged.
    """
    z = (mean_logprob - bundle.sentence_logprob_mean) / bundle.sentence_logprob_std
    shifted = (z - settings.sentence_zscore_offset) * settings.sentence_zscore_slope
    return 1.0 / (1.0 + math.exp(-max(min(shifted, 30.0), -30.0)))


def _sentence_reasons(
    score: signals_mod.SentenceScore,
    likelihood: float,
    essay_sentence_ppl_mean: float,
    settings: Settings,
) -> list[str]:
    """Concrete evidence strings -- always a sentence, never a bare score."""
    reasons: list[str] = []

    if score.token_count == 0:
        return reasons

    if likelihood >= settings.sentence_flag_threshold:
        reasons.append(
            f"Highly predictable to GPT-2 (perplexity {score.perplexity:.1f}), "
            "a pattern common in the AI-generated examples."
        )

    if essay_sentence_ppl_mean and score.perplexity < 0.6 * essay_sentence_ppl_mean:
        reasons.append(
            f"Perplexity {score.perplexity:.1f} sits well below this essay's "
            f"average of {essay_sentence_ppl_mean:.1f}."
        )

    markers = signals_mod.discourse_markers_in(score.sentence.text)
    if markers:
        listed = ", ".join(sorted(set(markers))[:3])
        reasons.append(f"Opens on a formulaic connective ({listed}).")

    if score.sentence.word_count >= 32:
        reasons.append(
            f"Unusually long and evenly built at {score.sentence.word_count} words."
        )

    return reasons


def _confidence(probability: float, settings: Settings) -> float:
    """How far from the decision boundary, normalised to 0-1.

    Each side of the boundary is scaled by its own width so an off-centre
    threshold still maps to a full 0-1 range instead of overflowing it.
    """
    threshold = settings.ai_decision_threshold
    if probability >= threshold:
        span = 1.0 - threshold
    else:
        span = threshold
    if span <= 0.0:
        return 1.0
    return round(min(abs(probability - threshold) / span, 1.0), 4)


def score_essay(text: str, settings: Settings | None = None) -> AnalyzeResponse:
    """Score an essay end to end and return probabilities, not a verdict.

    Both models are already in memory -- this reads the singletons, it never
    loads anything.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    gpt2 = get_gpt2()
    clf = get_classifier()

    signal_bundle = signals_mod.extract_signals(text, gpt2)
    # Built in the artifact's feature order, by name -- not signals.vector(),
    # which uses signals.FEATURE_NAMES. The model decides its own input layout.
    features_vector = np.asarray(
        [[float(signal_bundle.features[name]) for name in clf.feature_names]],
        dtype=float,
    )

    # predict_proba, never predict: predict() collapses the score to a 0/1
    # label and discards the confidence this whole design is built to report.
    # Column 1 is P(AI); load_classifier asserts classes_ == [0, 1].
    probability = float(clf.estimator.predict_proba(features_vector)[0][1])
    probability = min(max(probability, 0.0), 1.0)

    features = signal_bundle.features
    sentence_ppl_mean = float(features.get("sentence_perplexity_mean", 0.0))

    insights: list[SentenceInsight] = []
    flagged_count = 0
    for score in signal_bundle.sentence_scores[: settings.max_explained_sentences]:
        likelihood = _sentence_ai_likelihood(score.mean_logprob, clf, settings)
        flagged = (
            likelihood >= settings.sentence_flag_threshold and score.token_count > 0
        )
        flagged_count += int(flagged)
        insights.append(
            SentenceInsight(
                index=score.sentence.index,
                text=score.sentence.text,
                start_char=score.sentence.start,
                end_char=score.sentence.end,
                word_count=score.sentence.word_count,
                perplexity=round(score.perplexity, 4),
                mean_logprob=round(score.mean_logprob, 6),
                ai_likelihood=round(likelihood, 4),
                phrase=_sentence_phrase(likelihood, settings),
                flagged=flagged,
                reasons=_sentence_reasons(
                    score, likelihood, sentence_ppl_mean, settings
                ),
            )
        )

    summary = SignalSummary(
        perplexity=round(features["perplexity"], 4),
        mean_token_logprob=round(features["mean_token_logprob"], 6),
        std_token_logprob=round(features["std_token_logprob"], 6),
        mean_token_entropy=round(features["mean_token_entropy"], 6),
        burstiness=round(features["burstiness"], 6),
        sentence_length_mean=round(features["sentence_length_mean"], 4),
        sentence_length_std=round(features["sentence_length_std"], 4),
        sentence_length_cv=round(features["sentence_length_cv"], 6),
        perplexity_burstiness=round(features["perplexity_burstiness"], 6),
        type_token_ratio=round(features["type_token_ratio"], 6),
        hapax_ratio=round(features["hapax_ratio"], 6),
        avg_word_length=round(features["avg_word_length"], 4),
        stopword_ratio=round(features["stopword_ratio"], 6),
        punctuation_density=round(features["punctuation_density"], 6),
        comma_per_sentence=round(features["comma_per_sentence"], 4),
        discourse_marker_ratio=round(features["discourse_marker_ratio"], 6),
        contraction_ratio=round(features["contraction_ratio"], 6),
        # From the bundle, not the raw essay: the feature vector was computed
        # against normalised text, so recounting the input could disagree.
        word_count=signal_bundle.word_count,
        sentence_count=len(signal_bundle.sentences),
    )

    return AnalyzeResponse(
        overall_probability=probability,
        confidence_label=confidence_label(probability),
        label=_short_label(probability),
        confidence=_confidence(probability, settings),
        signals=summary,
        sentences=insights,
        flagged_sentence_count=flagged_count,
        top_features=_feature_contributions(
            clf.estimator, signal_bundle, clf.feature_names
        ),
        model_version=clf.version,
        processing_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )

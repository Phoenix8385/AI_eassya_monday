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
    # name -> fitted coefficient. The per-sentence flagging borrows these, so a
    # signal the model actually learned to weigh drives more of the flagging.
    feature_weights: dict[str, float]
    version: str
    sentence_logprob_mean: float
    sentence_logprob_std: float
    source_path: Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """Sample standard deviation; 0.0 when there is nothing to vary."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _extract_feature_weights(
    payload: Any, estimator: Any, feature_names: tuple[str, ...]
) -> dict[str, float]:
    """Read the trained coefficients, preferring the artifact's own mapping.

    ``train_classifier.py`` saves ``feature_weights`` directly. Older artifacts
    do not, so the coefficients are recovered from the fitted estimator instead
    -- same numbers, just unwrapped here rather than at training time.
    """
    if isinstance(payload, dict):
        stored = payload.get("feature_weights")
        if isinstance(stored, dict) and stored:
            return {str(k): float(v) for k, v in stored.items()}

    steps = getattr(estimator, "steps", None)
    final = steps[-1][1] if steps else estimator
    coefficients = getattr(final, "coef_", None)
    if coefficients is None:
        return {}

    flat = np.ravel(coefficients)
    if len(flat) != len(feature_names):
        return {}
    return {name: float(value) for name, value in zip(feature_names, flat)}


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
                "`python -m scripts.train_classifier` or point MODEL_PATH at an existing artifact."
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

        feature_weights = _extract_feature_weights(
            payload, estimator, tuple(feature_names)
        )
        if not feature_weights:
            logger.warning(
                "Artifact at %s exposes no feature weights; per-sentence "
                "flagging will weight its two signals equally instead of "
                "following the model's own coefficients.",
                path,
            )

        _bundle = ClassifierBundle(
            estimator=estimator,
            feature_names=tuple(feature_names),
            feature_weights=feature_weights,
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


# Which trained coefficient governs each per-sentence signal. Checked in
# order; the first name present in the artifact wins. Both lists describe the
# same underlying quantity the per-sentence deviation measures, so whichever
# the model was actually fitted on is the right weight to borrow.
_PERPLEXITY_SIGNAL_FEATURES = (
    "sentence_perplexity_std",
    "perplexity_burstiness",
    "perplexity",
)
_LENGTH_SIGNAL_FEATURES = (
    "sentence_length_std",
    "sentence_length_cv",
    "burstiness",
)

# Fixed templates, one per (signal, direction). Only the AI-indicative
# direction of each signal produces a reason: a sentence that is *less*
# predictable than its neighbours is evidence toward human authorship, and
# flagging it would invert the meaning of "flagged".
#
# Each opens on the same resemblance framing as the headline sentence, then
# names the specific signal. A per-sentence reason that read as a flat
# assertion ("this sentence IS predictable") would undercut the calibrated
# wording of the essay-level result sitting directly above it.
_REASON_TEMPLATES = {
    ("perplexity", "low"): (
        "This sentence resembles the AI-pattern examples in the training data "
        "— it's unusually predictable word-to-word compared to the rest of "
        "the essay."
    ),
    ("length", "high"): (
        "This sentence resembles the AI-pattern examples in the training data "
        "— it's unusually long and evenly built compared to the rest of the "
        "essay."
    ),
}

# Returned on every sentence when the essay is too short to localize against.
TOO_FEW_SENTENCES_REASON = (
    "Too few sentences in this essay to reliably localize signals"
)


class EssayTooShortError(ValueError):
    """The essay has no scoreable content.

    The API layer already enforces a minimum length, but scoring defends
    itself: an essay that survives validation and still yields no sentences
    (punctuation only, or a single unsegmentable fragment) must produce a clear
    error rather than a divide-by-zero deep in the statistics.
    """


@dataclass(frozen=True)
class SentenceDeviation:
    """Two-part within-essay anomaly score for one sentence.

    Deliberately separate from :func:`_sentence_ai_likelihood`, which answers a
    different question. That one asks "how AI-like is this sentence in absolute
    terms, against the training corpus". This one asks "does this sentence
    stand out from the other sentences in *this* essay" -- which is what makes
    a flag explainable to the person reading their own writing.
    """

    ppl_z: float  # negative = more predictable than this essay's norm
    length_z: float  # positive = longer than this essay's norm
    ppl_weighted: float  # |coef| x AI-direction deviation
    length_weighted: float
    combined: float  # the larger of the two -- the dominant signal
    dominant: str | None  # "perplexity" | "length" | None
    direction: str | None  # "low" | "high" | None


def _z_score(value: float, mean: float, std: float) -> float:
    """Z-score with a divide-by-zero guard.

    A near-constant essay (every sentence the same length, or GPT-2 finding
    every sentence equally predictable) has std ~ 0. That is informative in its
    own right -- it is what low burstiness *means*, and the essay-level
    features already capture it -- so it is not an error. It simply means no
    individual sentence deviates, hence zero.
    """
    if std <= 1e-9:
        return 0.0
    return (value - mean) / std


def _signal_weight(
    bundle: ClassifierBundle, candidates: tuple[str, ...]
) -> tuple[float, str | None]:
    """Magnitude of the model's own coefficient for a per-sentence signal.

    The point of borrowing the trained coefficient rather than picking a weight
    by hand: a signal the classifier learned to lean on drives more of the
    per-sentence flagging, so the flag traces back to the fitted model instead
    of a parallel heuristic that happens to sit next to it.
    """
    for name in candidates:
        weight = bundle.feature_weights.get(name)
        if weight is not None:
            return abs(float(weight)), name
    return 0.0, None


def _sentence_deviations(
    scores: list[signals_mod.SentenceScore],
    bundle: ClassifierBundle,
    settings: Settings,
) -> list[SentenceDeviation] | None:
    """Score every sentence against this essay's own distribution.

    Returns ``None`` when the essay has fewer than
    ``min_sentences_for_localization`` scoreable sentences. A standard
    deviation over two or three values is dominated by whichever sentence
    happens to be longest; z-scoring against it would manufacture confident
    numbers out of noise. Refusing to localize is the honest result, and the
    caller turns it into an explicit reason rather than a silent zero.

    Both weights are rescaled to average 1.0 across the two signals. That keeps
    ``combined`` on the z-scale -- a signal of average importance passes its
    z-score through unchanged, a more-important one amplifies it -- so the
    configured threshold stays interpretable as "about N standard deviations"
    no matter how large the raw coefficients happen to be.
    """
    scored = [s for s in scores if s.token_count > 0]
    if len(scored) < settings.min_sentences_for_localization:
        return None

    perplexities = [s.perplexity for s in scored]
    lengths = [float(s.sentence.word_count) for s in scored]

    ppl_mean, ppl_std = _mean(perplexities), _std(perplexities)
    len_mean, len_std = _mean(lengths), _std(lengths)

    ppl_weight, _ = _signal_weight(bundle, _PERPLEXITY_SIGNAL_FEATURES)
    len_weight, _ = _signal_weight(bundle, _LENGTH_SIGNAL_FEATURES)

    average = (ppl_weight + len_weight) / 2.0
    if average <= 1e-12:
        # No usable coefficients (e.g. a tree model, or an artifact predating
        # feature_weights). Fall back to equal weighting rather than silently
        # scoring everything zero.
        ppl_norm = len_norm = 1.0
    else:
        ppl_norm, len_norm = ppl_weight / average, len_weight / average

    deviations: list[SentenceDeviation] = []
    for score in scores:
        if score.token_count == 0:
            deviations.append(SentenceDeviation(0.0, 0.0, 0.0, 0.0, 0.0, None, None))
            continue

        ppl_z = _z_score(score.perplexity, ppl_mean, ppl_std)
        length_z = _z_score(float(score.sentence.word_count), len_mean, len_std)

        # Only the AI-indicative side of each signal contributes: predictable
        # (low perplexity) and long (high word count).
        ppl_weighted = ppl_norm * max(-ppl_z, 0.0)
        length_weighted = len_norm * max(length_z, 0.0)

        if ppl_weighted >= length_weighted:
            combined, dominant, direction = ppl_weighted, "perplexity", "low"
        else:
            combined, dominant, direction = length_weighted, "length", "high"

        if combined <= 0.0:
            dominant = direction = None

        deviations.append(
            SentenceDeviation(
                ppl_z=ppl_z,
                length_z=length_z,
                ppl_weighted=ppl_weighted,
                length_weighted=length_weighted,
                combined=combined,
                dominant=dominant,
                direction=direction,
            )
        )

    return deviations


def _reason_for(deviation: SentenceDeviation, settings: Settings) -> str | None:
    """Templated reason for the dominant signal, or None if it is too weak.

    The floor is on the *raw* z-score, not the weighted score: a coefficient
    large enough to push a 0.3-sigma wobble over the threshold must not be able
    to buy a confident-sounding sentence about a signal that barely moved.
    """
    if deviation.dominant is None or deviation.direction is None:
        return None

    raw_z = deviation.ppl_z if deviation.dominant == "perplexity" else deviation.length_z
    if abs(raw_z) < settings.sentence_min_z:
        return None

    return _REASON_TEMPLATES.get((deviation.dominant, deviation.direction))


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

    # Defence in depth. The router already rejects anything under
    # MIN_ESSAY_CHARS, but length in characters does not guarantee scoreable
    # content: punctuation-only input, or text that segments into nothing,
    # clears validation and would otherwise divide by zero further down.
    if not signal_bundle.sentences or not signal_bundle.sentence_scores:
        raise EssayTooShortError(
            "This text has no analysable sentences. Submit a longer essay in "
            "prose form."
        )
    if signal_bundle.word_count == 0:
        raise EssayTooShortError(
            "This text contains no words to analyse. Submit a longer essay in "
            "prose form."
        )

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

    # Computed once from the SAME extract_signals pass -- the per-sentence
    # perplexities were produced by the single forward sweep over the essay,
    # so nothing is re-tokenized or re-scored here. None means the essay was
    # too short to localize against its own variance.
    deviations = _sentence_deviations(signal_bundle.sentence_scores, clf, settings)
    localized = deviations is not None

    # A sentence being locally unusual inside an essay the classifier scored as
    # clearly human is not evidence of anything -- essays legitimately contain
    # one long or one plain sentence. The essay-level result gates the
    # per-sentence flags so the highlighting can never contradict the headline.
    essay_supports_flagging = probability >= settings.ai_decision_threshold

    insights: list[SentenceInsight] = []
    flagged_count = 0
    visible = signal_bundle.sentence_scores[: settings.max_explained_sentences]

    for position, score in enumerate(visible):
        likelihood = _sentence_ai_likelihood(score.mean_logprob, clf, settings)

        if not localized:
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
                    flagged=False,
                    reason=TOO_FEW_SENTENCES_REASON,
                    local_score=None,
                    ppl_z=None,
                    length_z=None,
                )
            )
            continue

        deviation = deviations[position]
        reason = _reason_for(deviation, settings)

        # Three gates, all required: the essay itself scored AI-leaning, the
        # coefficient-weighted deviation cleared the threshold, and a reason
        # survived the raw-z floor. The last one means a flag can never appear
        # with nothing to say about it.
        flagged = (
            score.token_count > 0
            and essay_supports_flagging
            and deviation.combined >= settings.sentence_flag_z
            and reason is not None
        )
        if not flagged:
            reason = None
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
                reason=reason,
                local_score=round(deviation.combined, 4),
                ppl_z=round(deviation.ppl_z, 4),
                length_z=round(deviation.length_z, 4),
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

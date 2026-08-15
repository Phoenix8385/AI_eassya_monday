"""Request/response contracts for the API.

The response models mirror exactly what ``services.classifier.score_essay``
builds -- that function returns :class:`AnalyzeResponse` itself, so the schema
and the producer cannot drift apart.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import get_settings

_settings = get_settings()

# Length bounds come from config, not from a literal baked into the schema.
EssayText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=_settings.min_essay_chars,
        max_length=_settings.max_essay_chars,
    ),
]

# A short tag, always shown alongside the probability and its sentence -- never
# on its own. It is a leaning, not a verdict, which is why both values carry
# "Likely".
Label = Literal["Likely AI", "Likely Human"]


class AnalyzeRequest(BaseModel):
    """Body of ``POST /analyze``."""

    model_config = ConfigDict(extra="forbid")

    essay: EssayText = Field(
        ...,
        description=(
            "The essay to analyse. Must be between "
            f"{_settings.min_essay_chars} and {_settings.max_essay_chars} "
            "characters after surrounding whitespace is trimmed."
        ),
    )


class SignalSummary(BaseModel):
    """Document-level signals, the same numbers the classifier scored on."""

    perplexity: float = Field(..., description="GPT-2 perplexity over the essay.")
    mean_token_logprob: float
    std_token_logprob: float
    mean_token_entropy: float = Field(
        ..., description="Mean entropy of GPT-2's next-token distribution."
    )
    burstiness: float = Field(
        ...,
        description="(sigma - mu) / (sigma + mu) over sentence lengths; "
        "human writing tends to sit higher.",
    )
    sentence_length_mean: float
    sentence_length_std: float
    sentence_length_cv: float
    perplexity_burstiness: float = Field(
        ..., description="Coefficient of variation of per-sentence perplexity."
    )
    type_token_ratio: float
    hapax_ratio: float
    avg_word_length: float
    stopword_ratio: float
    punctuation_density: float
    comma_per_sentence: float
    discourse_marker_ratio: float
    contraction_ratio: float
    word_count: int
    sentence_count: int


class SentenceInsight(BaseModel):
    """Per-sentence evidence backing the verdict."""

    index: int = Field(..., ge=0)
    text: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    perplexity: float
    mean_logprob: float
    ai_likelihood: float = Field(..., ge=0.0, le=1.0)
    phrase: str = Field(
        ...,
        description="How this sentence reads against the training examples, "
        "e.g. 'resembles human-written examples'. Always present, so a "
        "per-sentence score is never displayed as a bare number.",
    )
    flagged: bool
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable evidence, empty when nothing stood out.",
    )


class FeatureContribution(BaseModel):
    """A single feature's push toward or away from the 'AI' class."""

    name: str
    value: float
    contribution: float = Field(
        ..., description="Signed; positive pushes the essay toward 'AI'."
    )


class AnalyzeResponse(BaseModel):
    """Body of a successful ``POST /analyze``."""

    # These three are a set. The frontend shows them together; the probability
    # on its own is not a presentable result.
    overall_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw P(AI) from the classifier's predict_proba.",
    )
    confidence_label: str = Field(
        ...,
        description="A full sentence framing the probability as resemblance to "
        "training examples.",
    )
    label: Label = Field(..., description="Short tag shown beside the sentence.")

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Distance from the decision boundary."
    )
    signals: SignalSummary
    sentences: list[SentenceInsight]
    flagged_sentence_count: int = Field(..., ge=0)
    top_features: list[FeatureContribution]
    model_version: str
    processing_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["ok"]
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    error: str

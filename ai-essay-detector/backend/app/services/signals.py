"""Signal extraction: perplexity, burstiness and stylistic features.

Every function here is pure -- given the same text (and, where a language model
is needed, the same bundle) it returns the same numbers, and it never loads a
model, touches the network or reads config. The GPT-2 bundle arrives as an
argument; ``services.model_loader`` owns its lifetime.

Sentence splitting is deliberately regex-based rather than NLTK's punkt: punkt
needs a corpus download at runtime, which would make a fresh deploy fail on the
first request. The rules below cover the abbreviations that actually appear in
admissions essays.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

import torch

# The exact order the classifier was trained on. Appending is safe; reordering
# or removing invalidates every previously trained .joblib.
FEATURE_NAMES: tuple[str, ...] = (
    "perplexity",
    "mean_token_logprob",
    "std_token_logprob",
    "median_token_logprob",
    "mean_token_entropy",
    "burstiness",
    "sentence_length_mean",
    "sentence_length_std",
    "sentence_length_cv",
    "perplexity_burstiness",
    "sentence_perplexity_mean",
    "sentence_perplexity_std",
    "type_token_ratio",
    "hapax_ratio",
    "avg_word_length",
    "stopword_ratio",
    "punctuation_density",
    "comma_per_sentence",
    "discourse_marker_ratio",
    "contraction_ratio",
    "uppercase_ratio",
    "digit_ratio",
)

# Abbreviations that end in '.' without ending a sentence.
_ABBREVIATIONS = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "approx", "dept", "univ", "inc", "ltd", "co", "fig", "no", "vol",
    "al", "ph.d", "b.a", "m.a", "u.s", "u.k",
)
_ABBREV_SET = frozenset(_ABBREVIATIONS)

_SENTENCE_END = re.compile(r"([.!?]+[\"'”’\)\]]*)(\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_TOKEN_RE = re.compile(r"\S+")

_STOPWORDS = frozenset(
    """a an the and or but if while of to in on at by for with about against
    between into through during before after above below from up down out off
    over under again further then once here there all any both each few more
    most other some such no nor not only own same so than too very s t can will
    just don should now i me my myself we our ours ourselves you your yours he
    him his she her hers it its they them their what which who whom this that
    these those am is are was were be been being have has had having do does
    did doing as because until""".split()
)

# Connectives that LLM prose leans on far harder than student prose does.
_DISCOURSE_MARKERS = frozenset(
    """however moreover furthermore additionally consequently therefore thus
    nevertheless nonetheless meanwhile similarly conversely notably importantly
    ultimately overall subsequently accordingly hence whereas""".split()
)

_CONTRACTION_RE = re.compile(r"\b\w+['’](?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Sentence:
    """A sentence with its span in the original text."""

    index: int
    text: str
    start: int
    end: int
    word_count: int


@dataclass(frozen=True)
class SentenceScore:
    """Language-model evidence for one sentence."""

    sentence: Sentence
    perplexity: float
    mean_logprob: float
    token_count: int


@dataclass
class SignalBundle:
    """Everything one pass over an essay produced."""

    features: dict[str, float]
    # The NFKC-normalised text every offset and feature was computed against.
    # Callers must use this, not the raw input, or character spans will not line
    # up and word counts will disagree with the feature vector.
    normalized_text: str = ""
    word_count: int = 0
    sentences: list[Sentence] = field(default_factory=list)
    sentence_scores: list[SentenceScore] = field(default_factory=list)

    def vector(self) -> list[float]:
        """Features in :data:`FEATURE_NAMES` order, ready for the classifier."""
        return [float(self.features[name]) for name in FEATURE_NAMES]


# --------------------------------------------------------------------------
# Text segmentation
# --------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """NFKC-normalise and standardise newlines, preserving length semantics."""
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")


def _is_abbreviation(text: str, period_index: int) -> bool:
    """True when the '.' at ``period_index`` closes a known abbreviation."""
    start = period_index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    token = text[start:period_index].lower().rstrip(".")
    if not token:
        return False
    if token in _ABBREV_SET:
        return True
    # Single initials such as "J." in "J. R. Smith".
    return len(token) == 1 and token.isalpha()


def split_sentences(text: str) -> list[Sentence]:
    """Split ``text`` into sentences carrying their character spans."""
    sentences: list[Sentence] = []
    cursor = 0
    length = len(text)

    for match in _SENTENCE_END.finditer(text):
        end = match.end(1)
        # A '.' inside an abbreviation is not a boundary.
        if match.group(1) == "." and _is_abbreviation(text, match.start(1)):
            continue
        chunk = text[cursor:end]
        if chunk.strip():
            sentences.append(_make_sentence(len(sentences), text, cursor, end))
        cursor = match.end()

    if cursor < length and text[cursor:].strip():
        sentences.append(_make_sentence(len(sentences), text, cursor, length))

    return sentences


def _make_sentence(index: int, text: str, start: int, end: int) -> Sentence:
    """Build a Sentence, trimming surrounding whitespace off the span."""
    raw = text[start:end]
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    start += lead
    end -= trail
    body = text[start:end]
    return Sentence(
        index=index,
        text=body,
        start=start,
        end=end,
        word_count=len(_WORD_RE.findall(body)),
    )


def tokenize_words(text: str) -> list[str]:
    """Lowercased alphabetic word tokens."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def discourse_markers_in(text: str) -> list[str]:
    """The formulaic connectives present in ``text``, in order of appearance."""
    return [w for w in tokenize_words(text) if w in _DISCOURSE_MARKERS]


# --------------------------------------------------------------------------
# Perplexity
# --------------------------------------------------------------------------


def token_log_probabilities(
    text: str, bundle
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Per-token log-probability and predictive entropy under GPT-2.

    The whole essay is scored in a single sweep of overlapping windows rather
    than one forward pass per sentence -- for a 600-word essay that is one pass
    instead of ~30. Each window after the first re-reads ``stride`` tokens of
    context so that no token is ever scored without a left context.

    Returns ``(offsets, logprobs, entropies)`` aligned token-for-token; the very
    first token of the essay has no predecessor and is dropped from all three.
    """
    encoded = bundle.encode(text)
    ids: list[int] = list(encoded["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in encoded["offset_mapping"]]

    if len(ids) < 2:
        return [], [], []

    total = len(ids)
    window = bundle.max_window_tokens
    stride = max(1, min(bundle.stride_tokens, window - 1))

    logprobs = [math.nan] * total
    entropies = [math.nan] * total

    start = 0
    with torch.inference_mode():
        while start < total - 1:
            end = min(start + window, total)
            chunk = torch.tensor([ids[start:end]], device=bundle.device)

            logits = bundle.model(chunk).logits[0].float()
            log_probs = torch.log_softmax(logits, dim=-1)

            # log_probs[i] predicts token i+1, so drop the final row.
            predictive = log_probs[:-1]
            targets = chunk[0, 1:]
            chosen = predictive.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            entropy = -(predictive.exp() * predictive).sum(dim=-1)

            chosen_list = chosen.tolist()
            entropy_list = entropy.tolist()

            for i, (lp, ent) in enumerate(zip(chosen_list, entropy_list)):
                global_index = start + i + 1
                # Overlapped tokens keep their first (best-context) score.
                if math.isnan(logprobs[global_index]):
                    logprobs[global_index] = lp
                    entropies[global_index] = ent

            if end >= total:
                break
            start = end - stride

    keep = [i for i in range(total) if not math.isnan(logprobs[i])]
    return (
        [offsets[i] for i in keep],
        [logprobs[i] for i in keep],
        [entropies[i] for i in keep],
    )


def perplexity_from_logprobs(logprobs: list[float]) -> float:
    """exp(-mean log p). Clamped so a degenerate essay cannot return inf."""
    if not logprobs:
        return 0.0
    mean = sum(logprobs) / len(logprobs)
    return float(math.exp(min(-mean, 20.0)))


def score_sentences(
    sentences: list[Sentence],
    offsets: list[tuple[int, int]],
    logprobs: list[float],
) -> list[SentenceScore]:
    """Attribute already-computed token log-probs to their sentences.

    Tokens are matched by character offset, so this is a linear merge of two
    sorted sequences -- no re-tokenisation, no extra forward passes.
    """
    scores: list[SentenceScore] = []
    cursor = 0
    n_tokens = len(offsets)

    for sentence in sentences:
        # Skip tokens that ended before this sentence began.
        while cursor < n_tokens and offsets[cursor][1] <= sentence.start:
            cursor += 1

        i = cursor
        collected: list[float] = []
        while i < n_tokens and offsets[i][0] < sentence.end:
            collected.append(logprobs[i])
            i += 1

        if collected:
            mean = sum(collected) / len(collected)
            ppl = float(math.exp(min(-mean, 20.0)))
        else:
            mean, ppl = 0.0, 0.0

        scores.append(
            SentenceScore(
                sentence=sentence,
                perplexity=ppl,
                mean_logprob=mean,
                token_count=len(collected),
            )
        )

    return scores


# --------------------------------------------------------------------------
# Burstiness and style
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def burstiness_features(sentences: list[Sentence]) -> dict[str, float]:
    """Sentence-rhythm variation.

    ``burstiness`` is the standard (sigma - mu) / (sigma + mu) index. Human
    writing mixes long and short sentences and scores higher; LLM prose tends
    toward a uniform cadence and scores lower.
    """
    lengths = [float(s.word_count) for s in sentences if s.word_count > 0]
    mu, sigma = _mean(lengths), _std(lengths)
    denominator = sigma + mu

    return {
        "burstiness": (sigma - mu) / denominator if denominator else 0.0,
        "sentence_length_mean": mu,
        "sentence_length_std": sigma,
        "sentence_length_cv": (sigma / mu) if mu else 0.0,
    }


def sentence_perplexity_features(scores: list[SentenceScore]) -> dict[str, float]:
    """Spread of perplexity across sentences -- uniformity is an AI tell."""
    values = [s.perplexity for s in scores if s.token_count > 0]
    mu, sigma = _mean(values), _std(values)
    return {
        "sentence_perplexity_mean": mu,
        "sentence_perplexity_std": sigma,
        "perplexity_burstiness": (sigma / mu) if mu else 0.0,
    }


def stylistic_features(text: str, sentences: list[Sentence]) -> dict[str, float]:
    """Lexical and punctuation habits, all normalised to rates."""
    words = tokenize_words(text)
    n_words = len(words)
    n_sentences = max(len(sentences), 1)

    if n_words == 0:
        return {
            "type_token_ratio": 0.0,
            "hapax_ratio": 0.0,
            "avg_word_length": 0.0,
            "stopword_ratio": 0.0,
            "punctuation_density": 0.0,
            "comma_per_sentence": 0.0,
            "discourse_marker_ratio": 0.0,
            "contraction_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "digit_ratio": 0.0,
        }

    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    hapax = sum(1 for c in counts.values() if c == 1)
    stopwords = sum(1 for w in words if w in _STOPWORDS)
    discourse = sum(1 for w in words if w in _DISCOURSE_MARKERS)
    punctuation = sum(1 for ch in text if unicodedata.category(ch).startswith("P"))
    commas = text.count(",")
    contractions = len(_CONTRACTION_RE.findall(text))
    raw_tokens = _TOKEN_RE.findall(text)
    uppercase = sum(1 for t in raw_tokens if t.isupper() and len(t) > 1)
    digits = sum(1 for ch in text if ch.isdigit())

    return {
        "type_token_ratio": len(counts) / n_words,
        "hapax_ratio": hapax / n_words,
        "avg_word_length": sum(len(w) for w in words) / n_words,
        "stopword_ratio": stopwords / n_words,
        "punctuation_density": punctuation / n_words,
        "comma_per_sentence": commas / n_sentences,
        "discourse_marker_ratio": discourse / n_words,
        "contraction_ratio": contractions / n_words,
        "uppercase_ratio": uppercase / max(len(raw_tokens), 1),
        "digit_ratio": digits / max(len(text), 1),
    }


def perplexity_features(
    logprobs: list[float], entropies: list[float]
) -> dict[str, float]:
    """Document-level language-model statistics."""
    return {
        "perplexity": perplexity_from_logprobs(logprobs),
        "mean_token_logprob": _mean(logprobs),
        "std_token_logprob": _std(logprobs),
        "median_token_logprob": _median(logprobs),
        "mean_token_entropy": _mean(entropies),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def extract_signals(text: str, bundle) -> SignalBundle:
    """Run every extractor over ``text`` in one pass and return the bundle.

    ``bundle`` is a loaded ``GPT2Bundle``; this function never loads it.
    """
    normalized = normalize_text(text)
    sentences = split_sentences(normalized)

    offsets, logprobs, entropies = token_log_probabilities(normalized, bundle)
    sentence_scores = score_sentences(sentences, offsets, logprobs)

    features: dict[str, float] = {}
    features.update(perplexity_features(logprobs, entropies))
    features.update(burstiness_features(sentences))
    features.update(sentence_perplexity_features(sentence_scores))
    features.update(stylistic_features(normalized, sentences))

    # Guarantee the vector is complete even if an essay hit an empty branch.
    for name in FEATURE_NAMES:
        features.setdefault(name, 0.0)

    return SignalBundle(
        features=features,
        normalized_text=normalized,
        word_count=len(tokenize_words(normalized)),
        sentences=sentences,
        sentence_scores=sentence_scores,
    )

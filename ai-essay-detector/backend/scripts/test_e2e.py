"""Phase 9 end-to-end smoke test against a running API.

    python -m scripts.test_e2e                      # run every check
    python -m scripts.test_e2e --api-url http://localhost:8001
    python -m scripts.test_e2e --no-sentences       # verdicts only, no breakdown
    python -m scripts.test_e2e --refresh-fixtures   # re-pick essays from real data

Replaces pasting essays into the UI one at a time. Four fixed essays go through
``POST /analyze``; the script prints a PASS/FAIL line for each rather than raw
JSON, and exits non-zero if any gating check failed, so it doubles as a
"is anything obviously broken" gate before recording a demo.

**The essays are hardcoded, and their provenance is hardcoded with them.** Every
fixture carries the dataset, the row, and whether it was in the model's training
split -- because "the classifier got this right" means nothing if the essay was
one it was fitted on. Fixtures are not written by hand: ``--refresh-fixtures``
reproduces ``train_classifier.py``'s split exactly (same ``test_size`` and
``random_state``), takes essays from the held-out side, and rewrites the block
below in place. That keeps the claim "not used in training" checkable rather
than asserted.

**Not every check gates the exit code.** KNOWN_POLISHED is the case the model is
expected to get wrong -- see EVALUATION.md's per-source accuracy -- so it reports
as informative and never fails the run. Treating it as a failure would mean the
gate goes red on the one result that is working as documented.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT  # noqa: E402

# The backend's 400s and confidence sentences contain em-dashes. A Windows
# console defaults to cp1252 and would raise UnicodeEncodeError part-way
# through the report; replacing unmappable characters keeps the run alive.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DATA_DIR = BACKEND_ROOT / "app" / "data"
DEFAULT_CACHE = DATA_DIR / "features_cache.csv"
EVALUATION_MD = BACKEND_ROOT / "EVALUATION.md"
DEFAULT_API_URL = "http://localhost:8000"

# Must match train_classifier.py's final fit, or "held out" is not held out.
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42

# ==========================================================================
# FIXTURES -- regenerate with `python -m scripts.test_e2e --refresh-fixtures`
# ==========================================================================
# <!-- FIXTURES:start -->
KNOWN_HUMAN = {
    "text": (
        'Our robotics team lost the regional championship because I '
        'miscalculated a gear ratio. Not the team, me. I had checked it '
        'twice at eleven at night and both times I made the same arithmetic '
        'error, which is the thing nobody tells you about checking your own '
        'work. The drive train stalled in the second match and we watched '
        'the whole thing from behind the barrier. Nobody yelled at me. That '
        'was worse. Now I make somebody else check my numbers, always, even '
        'when it is embarrassing to ask, and especially when I am certain I '
        'am right.'
    ),
    "expect_label": "Likely Human",
    "provenance": {
        "dataset": "app/data/sample_essays.csv",
        "source": 'seed-human',
        "row": 3,
        "true_label": 0,
        "split": "held-out",
        "cache_row": 3,
    },
}

KNOWN_AI = {
    "text": (
        'My passion for scientific inquiry was ignited at an early age when '
        'I first encountered the wonders of the natural world. '
        'Additionally, participating in numerous research initiatives has '
        'allowed me to develop critical thinking skills that will serve me '
        'well in my future endeavours. Consequently, I have come to '
        'understand that true learning extends far beyond the confines of '
        'the classroom. Therefore, I am eager to continue exploring these '
        'questions in a rigorous academic environment that values both '
        'intellectual curiosity and practical application.'
    ),
    "expect_label": "Likely AI",
    "provenance": {
        "dataset": "app/data/sample_essays.csv",
        "source": 'seed-ai',
        "row": 9,
        "true_label": 1,
        "split": "held-out",
        "cache_row": 9,
    },
}

# No held-out candidate for this slot in the current feature cache.
# Build app/data/ai_polished_essays.csv, retrain, then re-run --refresh-fixtures.
KNOWN_POLISHED = {
    "text": None,
    "original_text": None,
    "expect_label": "Likely AI",
    "provenance": {
        "dataset": "app/data/ai_polished_essays.csv",
        "source": None,
        "row": None,
        "true_label": 1,
        "split": "unavailable",
    },
}
# <!-- FIXTURES:end -->

# Written by hand and never regenerated: these are not drawn from a dataset,
# they exist to hit the length guards.
#
# There are two of them because the backend rejects short input at two
# different layers, and only one of them is the Phase 7 check:
#
#   MIN_ESSAY_CHARS (250)  Pydantic, in models/schemas.py -> 422, and its
#                          message talks about characters, not words.
#   MIN_ESSAY_WORDS (50)   the /analyze handler -> 400, and its message names
#                          the actual word count so the caller knows how much
#                          more to write.
#
# A genuinely tiny essay never reaches the second one -- it is rejected on
# characters first. So TOO_SHORT is padded past 250 characters while staying
# under 50 words, which is the only way to actually exercise the word-count
# handler. TOO_SHORT_TINY covers the layer in front of it, so that a change
# making one guard swallow the other cannot pass unnoticed.
TOO_SHORT = {
    "text": (
        "This submission is deliberately too short for reliable analysis, "
        "although it has been padded with enough additional characters to clear "
        "the request body length check, so that the word count guard in the "
        "analyze handler is the validation layer that actually rejects it."
    ),
    "expect_status": 400,
    "provenance": {"dataset": "(hand-written)", "split": "n/a"},
}

TOO_SHORT_TINY = {
    "text": (
        "This essay is deliberately far too short to analyse properly, holding "
        "only about twenty words in total here."
    ),
    "expect_status": 422,
    "provenance": {"dataset": "(hand-written)", "split": "n/a"},
}


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

WIDTH = 78


def rule(char: str = "=") -> str:
    return char * WIDTH


def heading(title: str) -> None:
    print(f"\n{rule()}\n{title}\n{rule()}")


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        " ".join(str(text).split()),
        width=WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
    )


class Results:
    """Tally of check outcomes.

    ``gating`` is what decides the exit code. Informative and skipped checks are
    printed and counted but never fail the run.
    """

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.informative = 0
        self.skipped = 0
        self.failures: list[str] = []

    def gate(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}")
        if detail:
            print(wrap(detail, "         "))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(name)

    def note(self, name: str, detail: str = "") -> None:
        print(f"  [INFO] {name}")
        if detail:
            print(wrap(detail, "         "))
        self.informative += 1

    def skip(self, name: str, detail: str = "") -> None:
        print(f"  [SKIP] {name}")
        if detail:
            print(wrap(detail, "         "))
        self.skipped += 1


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


class ApiResult:
    """One /analyze response: status plus decoded body, however it came back."""

    def __init__(self, status: int | None, body: dict | None, transport_error: str | None):
        self.status = status
        self.body = body or {}
        self.transport_error = transport_error

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.transport_error is None

    @property
    def error_message(self) -> str:
        """The backend's own message, which is the useful one.

        Every non-2xx from this API uses ``{"error": ...}``; the FastAPI default
        ``{"detail": ...}`` is read as a fallback so a handler regression shows
        up as a wrong message rather than a blank one.
        """
        for key in ("error", "detail"):
            value = self.body.get(key)
            if isinstance(value, str) and value:
                return value
        return self.transport_error or "(no message in response body)"


def call_analyze(api_url: str, essay: str, timeout: float) -> ApiResult:
    """POST one essay. Never raises -- transport failures come back as a result."""
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/analyze",
        data=json.dumps({"essay": essay}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return ApiResult(response.status, json.load(response), None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": raw[:300]}
        return ApiResult(exc.code, body, None)
    except urllib.error.URLError as exc:
        return ApiResult(None, None, f"could not reach {api_url}: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return ApiResult(None, None, f"request to {api_url} failed: {exc}")


def check_api_up(api_url: str, timeout: float) -> str | None:
    """Return an error string if /health is not reachable and ready."""
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=timeout) as r:
            body = json.load(r)
    except Exception as exc:  # noqa: BLE001 - any failure means "not up"
        return (
            f"Cannot reach {api_url}/health ({exc}).\n"
            "Start the backend first:\n"
            "  python -m uvicorn app.main:app --port 8000"
        )
    if not body.get("model_loaded"):
        return f"{api_url} is up but reports model_loaded=false; check the boot logs."
    return None


# --------------------------------------------------------------------------
# Reporting one analysed essay
# --------------------------------------------------------------------------


def print_provenance(fixture: dict) -> None:
    p = fixture.get("provenance", {})
    bits = [f"{p.get('dataset', '?')}"]
    if p.get("row") is not None:
        bits.append(f"row {p['row']}")
    if p.get("source"):
        bits.append(f"source={p['source']}")
    if p.get("true_label") is not None:
        bits.append(f"true label={p['true_label']}")
    bits.append(f"split={p.get('split', '?')}")
    print(f"  provenance: {', '.join(bits)}")

    if p.get("split") == "train":
        print(
            wrap(
                "WARNING: this essay was in the model's training split, so a "
                "correct answer here is not evidence of generalisation. Re-run "
                "--refresh-fixtures once more data exists.",
                "  ",
            )
        )


def print_verdict(result: ApiResult) -> None:
    body = result.body
    print(f"  overall_probability : {body.get('overall_probability')}")
    print(f"  label               : {body.get('label')}")
    print("  confidence_label    :")
    print(wrap(body.get("confidence_label", "(missing)"), "      "))


def print_sentences(result: ApiResult) -> None:
    """Per-sentence breakdown, so the reasons can be read against the text.

    The script cannot judge whether a reason 'feels right' for its sentence --
    that read-through is still a human job. This just puts the two next to each
    other so it does not mean digging through raw JSON.
    """
    sentences = result.body.get("sentences") or []
    flagged = result.body.get("flagged_sentence_count", 0)
    print(f"\n  per-sentence breakdown ({len(sentences)} sentences, {flagged} flagged):")

    if not sentences:
        print("      (none returned)")
        return

    for s in sentences:
        local = s.get("local_score")
        mark = "FLAG" if s.get("flagged") else "    "
        score = "null" if local is None else f"{local:.4f}"
        print(f"\n    [{mark}] #{s.get('index')}  local_score={score}")
        print(wrap(s.get("text", ""), "           "))

        reason = s.get("reason")
        if reason:
            print(wrap(f"reason: {reason}", "           -> "))
        elif local is None:
            # Should not happen: the backend fills TOO_FEW_SENTENCES_REASON on
            # every sentence when it skips localization.
            print("           -> reason: (none given for an unscored sentence)")

    if any(s.get("local_score") is None for s in sentences):
        print(
            wrap(
                "local_score=null means the essay had too few sentences for the "
                "backend to z-score against its own variance "
                "(MIN_SENTENCES_FOR_LOCALIZATION, default 4). Not an error.",
                "      ",
            )
        )


# --------------------------------------------------------------------------
# EVALUATION.md
# --------------------------------------------------------------------------


def read_polished_accuracy(path: Path) -> tuple[int, float] | None:
    """Pull the ai_polished row out of EVALUATION.md's per-source table.

    Read rather than hardcoded on purpose: the number is regenerated by
    ``scripts.generate_evaluation`` on every retrain, and a stale figure baked
    into this script would be exactly the inconsistency it is meant to catch.
    Table row format: ``| `ai_polished` | 40 | 27/40 | 0.6750 |``
    """
    if not path.exists():
        return None
    row = re.compile(
        r"^\|\s*`?([\w.-]*polish[\w.-]*)`?\s*\|\s*(\d+)\s*\|[^|]*\|\s*([0-9.]+)\s*\|",
        re.MULTILINE | re.IGNORECASE,
    )
    match = row.search(path.read_text(encoding="utf-8"))
    if not match:
        return None
    try:
        return int(match.group(2)), float(match.group(3))
    except ValueError:
        return None


def polished_context(path: Path) -> str:
    """The sentence explaining why a 'wrong' polished result is expected."""
    found = read_polished_accuracy(path)
    if found is None:
        return (
            "EVALUATION.md is not present (or has no ai_polished row), so there "
            "is no measured figure to read this against. Generate it with "
            "`python -m scripts.generate_evaluation` -- its per-source table is "
            "where the polished-subset accuracy comes from. Whatever that number "
            "turns out to be, this essay is exactly the kind of case it "
            "represents: human writing whose surface was machine-edited, which "
            "is the hardest case for this detector and the one most likely to "
            "arrive in real admissions reading."
        )
    n, accuracy = found
    return (
        f"EVALUATION.md reports {accuracy:.0%} accuracy on the ai_polished "
        f"subset ({n} essays) -- this is exactly the kind of case that number "
        f"represents. Roughly {1 - accuracy:.0%} of these are expected to come "
        "back on the wrong side, so a 'Likely Human' result here is the "
        "documented behaviour of the model, not a bug. The ideas, structure and "
        "voice are human; only the surface was machine-touched, and that is what "
        "this detector is weakest at."
    )


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def run_labelled_essay(
    title: str, fixture: dict, api_url: str, timeout: float, show_sentences: bool, results: Results
) -> ApiResult | None:
    """Run one essay expected to land on a particular side. Gating."""
    heading(title)

    if not fixture.get("text"):
        results.skip(
            f"{title}: no fixture available",
            f"Expected an essay from {fixture['provenance']['dataset']}, which "
            "does not exist yet. Build it, then re-run with --refresh-fixtures.",
        )
        return None

    print_provenance(fixture)
    words = len(fixture["text"].split())
    print(f"  length: {words} words, {len(fixture['text'])} chars")

    result = call_analyze(api_url, fixture["text"], timeout)
    if not result.ok:
        results.gate(
            f"{title}: request succeeded",
            False,
            f"HTTP {result.status}: {result.error_message}",
        )
        return None

    print()
    print_verdict(result)

    expected = fixture["expect_label"]
    actual = result.body.get("label")
    results.gate(
        f"{title}: label == {expected!r}",
        actual == expected,
        "" if actual == expected else f"got {actual!r} instead.",
    )

    if show_sentences:
        print_sentences(result)
    return result


def run_polished(
    fixture: dict, api_url: str, timeout: float, show_sentences: bool, results: Results
) -> None:
    """KNOWN_POLISHED. Informative only -- a 'wrong' answer here is expected."""
    heading("KNOWN_POLISHED -- AI-polished human essay (informative, never gates)")

    context = polished_context(EVALUATION_MD)

    if not fixture.get("text"):
        results.skip(
            "KNOWN_POLISHED: no fixture available",
            f"Needs {fixture['provenance']['dataset']}. Build it with "
            "`python -m scripts.build_polished_set`, retrain, then re-run with "
            "--refresh-fixtures.",
        )
        print()
        print(wrap(context, "      "))
        return

    print_provenance(fixture)
    result = call_analyze(api_url, fixture["text"], timeout)
    if not result.ok:
        results.gate(
            "KNOWN_POLISHED: request succeeded",
            False,
            f"HTTP {result.status}: {result.error_message}",
        )
        return

    print()
    print_verdict(result)

    actual = result.body.get("label")
    expected = fixture["expect_label"]
    agrees = actual == expected
    results.note(
        f"KNOWN_POLISHED: label is {actual!r} (true label is AI-polished, "
        f"so {expected!r} would be 'correct')",
        "This check does not gate the exit code."
        + ("" if agrees else " The model called this one the other way."),
    )
    print()
    print(wrap(context, "      "))

    original = fixture.get("original_text")
    if original:
        print("\n  --- same essay BEFORE polishing, for comparison ---")
        before = call_analyze(api_url, original, timeout)
        if before.ok:
            print(
                f"  original (true label: human) -> "
                f"{before.body.get('label')} at "
                f"p={before.body.get('overall_probability'):.4f}"
            )
            print(
                f"  polished (labelled AI)       -> "
                f"{result.body.get('label')} at "
                f"p={result.body.get('overall_probability'):.4f}"
            )
            delta = (
                result.body.get("overall_probability", 0)
                - before.body.get("overall_probability", 0)
            )
            print(
                wrap(
                    f"Polishing moved the probability by {delta:+.4f}. That shift "
                    "is the thing worth reading: it is how much of the signal is "
                    "surface style rather than authorship.",
                    "      ",
                )
            )
        else:
            print(f"  original essay request failed: HTTP {before.status}: {before.error_message}")

    if show_sentences:
        print_sentences(result)


def run_too_short(fixture: dict, api_url: str, timeout: float, results: Results) -> None:
    """TOO_SHORT must produce the Phase 7 word-count 400, not a 500 or a crash."""
    heading("TOO_SHORT -- word-count guard (expects HTTP 400)")

    words = len(fixture["text"].split())
    print(f"  length: {words} words, {len(fixture['text'])} chars")
    print(wrap(fixture["text"], "    "))
    print(
        wrap(
            "Padded past MIN_ESSAY_CHARS (250) on purpose: under that, Pydantic "
            "rejects it on characters and the word-count handler never runs.",
            "    ",
        )
    )

    result = call_analyze(api_url, fixture["text"], timeout)

    if result.transport_error:
        results.gate("TOO_SHORT: request reached the API", False, result.transport_error)
        return

    print(f"\n  HTTP {result.status}")
    print("  error message:")
    print(wrap(result.error_message, "      "))

    results.gate(
        "TOO_SHORT: status is 400 (not 500, not a crash)",
        result.status == 400,
        ""
        if result.status == 400
        else f"got {result.status}. A 422 here means the char bound fired first "
        "and the word-count handler was never reached.",
    )

    # A 400 alone is not enough -- the point of the Phase 7 handler is that the
    # caller is told how many words they actually wrote, so the message has to
    # carry the count. A generic failure with the right status still fails.
    message = result.error_message.lower()
    mentions_words = "word" in message
    mentions_count = str(words) in result.error_message or bool(
        re.search(r"\b\d+\s+words?\b", message)
    )
    results.gate(
        "TOO_SHORT: message names the word count",
        mentions_words and mentions_count,
        ""
        if (mentions_words and mentions_count)
        else "expected the word-count message from the /analyze handler "
        f"(e.g. 'Essay is {words} words - at least 50 words are needed...'), "
        "not a generic or character-bound error.",
    )


def run_too_short_tiny(fixture: dict, api_url: str, timeout: float, results: Results) -> None:
    """The layer in front: a genuinely tiny essay is rejected on characters.

    Checked so that the two guards cannot quietly collapse into one. If this
    ever returns 400 with a word-count message, MIN_ESSAY_CHARS stopped firing;
    if TOO_SHORT returns 422, the char bound grew and swallowed the handler.
    """
    heading("TOO_SHORT_TINY -- character guard in front of it (expects HTTP 422)")

    print(f"  length: {len(fixture['text'].split())} words, {len(fixture['text'])} chars")
    print(wrap(fixture["text"], "    "))

    result = call_analyze(api_url, fixture["text"], timeout)
    if result.transport_error:
        results.gate("TOO_SHORT_TINY: request reached the API", False, result.transport_error)
        return

    print(f"\n  HTTP {result.status}")
    print("  error message:")
    print(wrap(result.error_message, "      "))

    results.gate(
        "TOO_SHORT_TINY: rejected as a client error, not a 500 or a crash",
        result.status in (400, 422),
        "" if result.status in (400, 422) else f"got {result.status}.",
    )
    results.gate(
        "TOO_SHORT_TINY: status is 422 (char bound fired before the handler)",
        result.status == 422,
        ""
        if result.status == 422
        else f"got {result.status}; MIN_ESSAY_CHARS no longer rejects this first.",
    )


# --------------------------------------------------------------------------
# --refresh-fixtures
# --------------------------------------------------------------------------


def refresh_fixtures(cache_path: Path, script_path: Path, test_size: float, seed: int) -> int:
    """Re-pick the three essay fixtures from the model's held-out split.

    Reproduces ``train_classifier.py``'s ``train_test_split`` over the cached
    feature rows, then maps the held-out rows back to their source CSV through
    the cached ``_text_excerpt``. Matching on the excerpt rather than assuming
    row alignment matters: extraction drops unusable essays, so cache row *i* is
    not necessarily dataset row *i*.
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(f"--refresh-fixtures needs pandas/numpy/scikit-learn: {exc}")

    if not cache_path.exists():
        raise SystemExit(
            f"No feature cache at {cache_path}. Train first:\n"
            "  python -m scripts.train_classifier"
        )

    cache = pd.read_csv(cache_path)
    if "_text_excerpt" not in cache.columns:
        raise SystemExit(
            f"{cache_path.name} has no _text_excerpt column, so held-out rows "
            "cannot be mapped back to their essays. Re-run:\n"
            "  python -m scripts.train_classifier --refresh-cache"
        )

    y = cache["label"].to_numpy(dtype=int)
    stratify = y if np.bincount(y).min() >= 2 else None
    _, held_out = train_test_split(
        np.arange(len(cache)), test_size=test_size, random_state=seed, stratify=stratify
    )
    held_out = sorted(int(i) for i in held_out)
    print(f"Held-out rows (test_size={test_size}, seed={seed}): {held_out}")

    # Every dataset CSV except the cache itself, same set train_classifier reads.
    sources = {
        p.name: pd.read_csv(p)
        for p in sorted(DATA_DIR.glob("*.csv"))
        if p.name != cache_path.name
    }

    def find_text(excerpt: str) -> tuple[str, int, str, str | None] | None:
        """Locate the full essay behind a cached excerpt."""
        key = " ".join(str(excerpt).split())[:120]
        if not key:
            return None
        for name, frame in sources.items():
            if "text" not in frame.columns:
                continue
            for row_index, value in enumerate(frame["text"].astype(str)):
                if " ".join(value.split()).startswith(key):
                    original = None
                    if "original_text" in frame.columns:
                        candidate = frame.iloc[row_index].get("original_text")
                        if isinstance(candidate, str) and candidate.strip():
                            original = candidate
                    return name, row_index, value, original
        return None

    def pick(want_label: int, prefer: tuple[str, ...], require_original: bool = False):
        """First held-out row matching the label, preferring certain sources."""
        candidates = []
        for i in held_out:
            if int(cache.loc[i, "label"]) != want_label:
                continue
            found = find_text(cache.loc[i, "_text_excerpt"])
            if found is None:
                continue
            name, row_index, text, original = found
            if require_original and not original:
                continue
            source = str(cache.loc[i, "source"])
            rank = next(
                (r for r, p in enumerate(prefer) if p in source.lower()), len(prefer)
            )
            candidates.append((rank, i, name, row_index, text, original, source))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0]

    human = pick(0, ("daigt",))
    ai = pick(1, ("daigt",))
    polished = pick(1, ("polish",), require_original=True)

    for name, chosen in (("KNOWN_HUMAN", human), ("KNOWN_AI", ai), ("KNOWN_POLISHED", polished)):
        if chosen is None:
            print(f"  {name}: no held-out candidate found")
        else:
            print(f"  {name}: {chosen[2]} row {chosen[3]} (source={chosen[6]}, cache row {chosen[1]})")

    block = "\n\n".join(
        [
            render_fixture("KNOWN_HUMAN", human, "Likely Human", 0, "app/data/daigt_essays.csv"),
            render_fixture("KNOWN_AI", ai, "Likely AI", 1, "app/data/daigt_essays.csv"),
            render_fixture(
                "KNOWN_POLISHED",
                polished,
                "Likely AI",
                1,
                "app/data/ai_polished_essays.csv",
                with_original=True,
            ),
        ]
    )

    text = script_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(# <!-- FIXTURES:start -->\n).*?(# <!-- FIXTURES:end -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit("Could not find the FIXTURES markers in this file.")
    script_path.write_text(
        pattern.sub(lambda m: m.group(1) + block + "\n" + m.group(2), text), encoding="utf-8"
    )
    print(f"\nRewrote the FIXTURES block in {script_path.name}.")
    return 0


def render_fixture(
    name: str,
    chosen,
    expect_label: str,
    true_label: int,
    wanted_dataset: str,
    with_original: bool = False,
) -> str:
    """Emit one fixture as Python source for the regenerated block.

    ``wanted_dataset`` is recorded even when nothing was found, so the skip
    message can still name the file to build rather than saying "not found".
    """
    if chosen is None:
        original_line = '    "original_text": None,\n' if with_original else ""
        return (
            f"# No held-out candidate for this slot in the current feature cache.\n"
            f"# Build {wanted_dataset}, retrain, then re-run --refresh-fixtures.\n"
            f"{name} = {{\n"
            f'    "text": None,\n'
            f"{original_line}"
            f'    "expect_label": "{expect_label}",\n'
            f'    "provenance": {{\n'
            f'        "dataset": "{wanted_dataset}",\n'
            f'        "source": None,\n'
            f'        "row": None,\n'
            f'        "true_label": {true_label},\n'
            f'        "split": "unavailable",\n'
            f"    }},\n"
            f"}}"
        )

    _, cache_row, dataset, row_index, text, original, source = chosen
    lines = [f"{name} = {{", "    \"text\": (", _as_source_string(text, "        "), "    ),"]
    if with_original:
        if original:
            lines += ["    \"original_text\": (", _as_source_string(original, "        "), "    ),"]
        else:
            lines.append('    "original_text": None,')
    lines += [
        f'    "expect_label": "{expect_label}",',
        '    "provenance": {',
        f'        "dataset": "app/data/{dataset}",',
        f'        "source": {source!r},',
        f'        "row": {row_index},',
        f'        "true_label": {true_label},',
        f'        "split": "held-out",',
        f'        "cache_row": {cache_row},',
        "    },",
        "}",
    ]
    return "\n".join(lines)


def _as_source_string(text: str, indent: str) -> str:
    """Wrap an essay into adjacent string literals that fit in a source file."""
    flat = " ".join(str(text).split())
    lines = textwrap.wrap(flat, width=WIDTH - len(indent) - 4) or [""]
    out = []
    for i, line in enumerate(lines):
        suffix = " " if i < len(lines) - 1 else ""
        out.append(f"{indent}{(line + suffix)!r}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the running API.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout (s).")
    parser.add_argument(
        "--no-sentences",
        action="store_true",
        help="Skip the per-sentence breakdown and print verdicts only.",
    )
    parser.add_argument(
        "--refresh-fixtures",
        action="store_true",
        help="Re-pick the essay fixtures from the model's held-out split and "
        "rewrite the block at the top of this file.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--test-size", type=float, default=SPLIT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if args.refresh_fixtures:
        return refresh_fixtures(
            args.cache, Path(__file__).resolve(), args.test_size, args.seed
        )

    print(rule())
    print("Phase 9 end-to-end check")
    print(rule())
    print(f"API: {args.api_url}")

    problem = check_api_up(args.api_url, timeout=min(args.timeout, 15.0))
    if problem:
        print(f"\n{problem}")
        return 2

    print("Health: ok, models loaded.")

    results = Results()
    show = not args.no_sentences

    run_labelled_essay(
        "KNOWN_HUMAN -- human-written essay", KNOWN_HUMAN, args.api_url, args.timeout, show, results
    )
    run_labelled_essay(
        "KNOWN_AI -- AI-generated essay", KNOWN_AI, args.api_url, args.timeout, show, results
    )
    run_polished(KNOWN_POLISHED, args.api_url, args.timeout, show, results)
    run_too_short(TOO_SHORT, args.api_url, args.timeout, results)
    run_too_short_tiny(TOO_SHORT_TINY, args.api_url, args.timeout, results)

    # --- Summary ----------------------------------------------------------
    heading("SUMMARY")
    print(f"  gating checks passed : {results.passed}")
    print(f"  gating checks failed : {results.failed}")
    print(f"  informative (never gate) : {results.informative}")
    print(f"  skipped (fixture unavailable) : {results.skipped}")

    if results.failures:
        print("\n  failed:")
        for name in results.failures:
            print(f"    - {name}")

    if results.skipped:
        print(
            wrap(
                "Skipped checks are missing data, not failures, so they do not "
                "affect the exit code -- but the run is weaker than it looks "
                "until they are filled in.",
                "  ",
            )
        )

    if show:
        print(
            wrap(
                "The per-sentence reasons above still need one human read-through: "
                "this script checks that a reason exists and is attached to the "
                "right sentence, not that it is a sensible explanation of it.",
                "  ",
            )
        )

    print()
    if results.failed:
        print(f"RESULT: FAILED ({results.failed} gating check(s))")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the AI-polished hard-case set from human essays.

    python -m scripts.build_polished_set              # 40 essays, calls the API
    python -m scripts.build_polished_set --dry-run    # no API calls, no cost
    python -m scripts.build_polished_set --resume-only # rebuild CSV from cache

Takes human-written essays from daigt_essays.csv, has Claude lightly polish
each one, and labels the result as AI. These are the *hard* cases: the ideas,
structure and voice are human, only the surface has been machine-touched. A
detector that only catches wholly-generated text will fail here.

Three things this script does deliberately:

**Three rotating prompts, not one.** A single fixed prompt makes every polished
essay share one model's response to one instruction, so they cluster tightly in
feature space. A classifier learns that cluster and scores well on an
artificially easy problem. Rotating three prompts spreads the polish styles out
and keeps the evaluation honest.

**Resumable.** Every completed essay is appended to a JSONL cache keyed by its
source row index before the next API call starts. Re-running skips anything
already cached, so an interrupted run costs nothing to resume and the CSV can
be rebuilt from cache with no API calls at all.

**Failures are logged, not fatal.** An essay that fails all retry attempts is
written to failures.log with the reason; the batch continues. One bad essay
should not cost you the other 39.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow `python scripts/build_polished_set.py` as well as `-m scripts...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("build_polished_set")
for _noisy in ("httpx", "httpcore", "anthropic"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DATA_DIR = BACKEND_ROOT / "app" / "data"
DEFAULT_SOURCE = DATA_DIR / "daigt_essays.csv"
DEFAULT_OUT = DATA_DIR / "ai_polished_essays.csv"
DEFAULT_CACHE = DATA_DIR / "polished_cache.jsonl"
DEFAULT_FAILURES = DATA_DIR / "failures.log"

DEFAULT_MODEL = "claude-opus-5"
SAMPLE_SIZE = 40
RANDOM_STATE = 42

# Keep the model from wrapping the essay in commentary. Prefill is not
# available on current models, so the instruction carries this.
SYSTEM_PROMPT = (
    "You revise student essays. Return only the revised essay text: no "
    "preamble, no explanation of your changes, no markdown formatting, no "
    "surrounding quotation marks. Preserve the author's paragraph breaks."
)

# Rotated one-third each across the sample. Ordered; the variant for essay i is
# POLISH_PROMPTS[i % 3], so the assignment is deterministic and reproducible.
POLISH_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "light_grammar_flow",
        "Lightly polish this essay's grammar and flow without changing its "
        "ideas or structure.",
    ),
    (
        "sentence_variety_word_choice",
        "Improve the sentence variety and word choice in this essay while "
        "preserving its meaning.",
    ),
    (
        "grammar_phrasing_keep_voice",
        "Fix grammar and awkward phrasing in this essay, keep the voice "
        "similar.",
    ),
)


class PolishError(RuntimeError):
    """An essay could not be polished; carries a short reason for the log."""


@dataclass(frozen=True)
class Job:
    """One essay to polish."""

    index: int  # row index in the source CSV -- the cache key
    text: str
    variant: str
    instruction: str


# --------------------------------------------------------------------------
# Cache and failure log
# --------------------------------------------------------------------------


def load_cache(path: Path) -> dict[int, dict]:
    """Read the JSONL cache into {source_index: record}.

    A truncated final line (killed mid-write) is dropped rather than fatal.
    """
    if not path.exists():
        return {}

    records: dict[int, dict] = {}
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records[int(record["index"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                skipped += 1

    if skipped:
        logger.warning("Ignored %d unreadable cache line(s) in %s", skipped, path.name)
    return records


def append_cache(path: Path, record: dict) -> None:
    """Append one record and flush to disk immediately.

    Written before the next API call starts, so a crash loses at most the essay
    currently in flight.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def log_failure(path: Path, index: int, variant: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\tindex={index}\tvariant={variant}\t{reason}\n")
        handle.flush()


# --------------------------------------------------------------------------
# The API call
# --------------------------------------------------------------------------


def build_client(max_retries: int = 0):
    """Construct the Anthropic client.

    ``max_retries=0`` is deliberate: this script owns retrying. Left at the
    SDK default of 2, the two layers multiply and a rate-limited essay would
    make up to 9 requests instead of 3.

    Credentials resolve from the environment: ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile.
    """
    import anthropic

    return anthropic.Anthropic(max_retries=max_retries)


def _extract_text(message) -> str:
    """Join the text blocks of a response, ignoring thinking blocks."""
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()


def polish_once(client, job: Job, model: str, max_tokens: int) -> str:
    """One API call. Raises PolishError for outcomes that must not be retried."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": f"{job.instruction}\n\n---\n\n{job.text}",
            }
        ],
    )

    # A safety decline returns HTTP 200 -- check before reading content.
    if message.stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        raise PolishError(f"model refused (category={category})")

    text = _extract_text(message)
    if not text:
        raise PolishError(f"empty response (stop_reason={message.stop_reason})")
    if message.stop_reason == "max_tokens":
        raise PolishError("response truncated at max_tokens; raise --max-tokens")

    return text


def polish_with_retry(
    client, job: Job, model: str, max_tokens: int, max_attempts: int, base_delay: float
) -> str:
    """Call the API, retrying transient failures with exponential backoff.

    Retries rate limits, 5xx, timeouts and connection errors. Does not retry
    4xx client errors or refusals -- those fail the same way every time, so
    retrying only burns quota.
    """
    import anthropic

    last_reason = "unknown"

    for attempt in range(1, max_attempts + 1):
        try:
            return polish_once(client, job, model, max_tokens)

        except anthropic.RateLimitError as exc:
            last_reason = f"rate limited: {exc}"
            # Honour the server's own backoff hint when it sends one.
            header = getattr(getattr(exc, "response", None), "headers", {}) or {}
            try:
                delay = float(header.get("retry-after", 0)) or None
            except (TypeError, ValueError):
                delay = None

        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            last_reason = f"connection/timeout: {type(exc).__name__}: {exc}"
            delay = None

        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:
                # 400/401/403/404 -- deterministic, retrying cannot help.
                raise PolishError(
                    f"non-retryable API error {exc.status_code}: {exc}"
                ) from exc
            last_reason = f"server error {exc.status_code}: {exc}"
            delay = None

        except PolishError:
            raise  # refusal / empty / truncated -- deterministic

        if attempt == max_attempts:
            break

        wait = delay if delay else base_delay * (2 ** (attempt - 1))
        wait += random.uniform(0, 0.5)  # jitter, so parallel runs don't sync up
        logger.warning(
            "  index=%d attempt %d/%d failed (%s); retrying in %.1fs",
            job.index,
            attempt,
            max_attempts,
            last_reason,
            wait,
        )
        time.sleep(wait)

    raise PolishError(f"all {max_attempts} attempts failed -- last: {last_reason}")


def polish_dry_run(job: Job) -> str:
    """Offline stand-in used by --dry-run.

    Applies a crude deterministic edit so the pipeline, cache, rotation and CSV
    can be exercised without spending money. Never use its output as data.
    """
    time.sleep(0.01)
    return f"[DRY RUN {job.variant}] {job.text}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def select_jobs(source: Path, sample_size: int, seed: int) -> list[Job]:
    """Sample human essays reproducibly and assign a rotating prompt variant."""
    if not source.exists():
        raise SystemExit(
            f"No source dataset at {source}. Build it first:\n"
            "  kaggle competitions download -c llm-detect-ai-generated-text\n"
            "  unzip llm-detect-ai-generated-text.zip -d app/data/raw\n"
            "  python -m scripts.prepare_daigt"
        )

    frame = pd.read_csv(source)

    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise SystemExit(
            f"{source} is missing column(s) {sorted(missing)}; found "
            f"{list(frame.columns)}. Run scripts/prepare_daigt.py first."
        )

    human = frame[frame["label"] == 0]
    logger.info("%s: %d rows, %d human (label=0)", source.name, len(frame), len(human))

    if len(human) < sample_size:
        raise SystemExit(
            f"Need {sample_size} human essays, found only {len(human)}. "
            "Check that prepare_daigt.py picked up the labelled file."
        )

    # random_state makes the sample reproducible across runs, so the cache keys
    # stay valid and a resumed run works on the same 40 essays.
    sampled = human.sample(n=sample_size, random_state=seed)

    return [
        Job(
            index=int(row_index),
            text=str(row["text"]),
            variant=POLISH_PROMPTS[position % len(POLISH_PROMPTS)][0],
            instruction=POLISH_PROMPTS[position % len(POLISH_PROMPTS)][1],
        )
        for position, (row_index, row) in enumerate(sampled.iterrows())
    ]


def write_output(cache: dict[int, dict], out: Path) -> int:
    """Write every cached record to the output CSV."""
    if not cache:
        logger.error("Cache is empty; nothing to write.")
        return 0

    rows = [
        {
            "text": record["text"],
            "original_text": record["original_text"],
            "label": 1,
            "source": "ai_polished",
            "prompt_variant": record["prompt_variant"],
        }
        for _, record in sorted(cache.items())
    ]

    frame = pd.DataFrame(rows, columns=["text", "original_text", "label", "source", "prompt_variant"])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows, columns = %s)", out, len(frame), list(frame.columns))
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the pipeline with a local stub instead of the API. "
        "Costs nothing; the output is NOT usable training data.",
    )
    parser.add_argument(
        "--resume-only",
        action="store_true",
        help="Rebuild the CSV from the cache without making any API calls.",
    )
    args = parser.parse_args()

    jobs = select_jobs(args.source, args.sample_size, args.seed)
    cache = load_cache(args.cache)

    variant_counts: dict[str, int] = {}
    for job in jobs:
        variant_counts[job.variant] = variant_counts.get(job.variant, 0) + 1
    logger.info("Prompt rotation across %d essays: %s", len(jobs), variant_counts)

    pending = [job for job in jobs if job.index not in cache]
    logger.info(
        "Cache holds %d record(s); %d essay(s) still to do.", len(cache), len(pending)
    )

    if args.resume_only:
        logger.info("--resume-only: skipping all API calls.")
        write_output(cache, args.out)
        return

    if not pending:
        logger.info("Nothing pending -- everything is already cached.")
        write_output(cache, args.out)
        return

    client = None if args.dry_run else build_client()
    if args.dry_run:
        logger.warning("DRY RUN: using a local stub. Output is not real data.")
    else:
        logger.info("Model: %s (max_tokens=%d)", args.model, args.max_tokens)

    succeeded = failed = 0
    for position, job in enumerate(pending, start=1):
        logger.info(
            "[%d/%d] index=%d variant=%s (%d words)",
            position,
            len(pending),
            job.index,
            job.variant,
            len(job.text.split()),
        )
        try:
            polished = (
                polish_dry_run(job)
                if args.dry_run
                else polish_with_retry(
                    client,
                    job,
                    args.model,
                    args.max_tokens,
                    args.max_attempts,
                    args.base_delay,
                )
            )
        except PolishError as exc:
            failed += 1
            logger.error("  index=%d FAILED: %s", job.index, exc)
            log_failure(args.failures, job.index, job.variant, str(exc))
            continue
        except Exception as exc:  # unexpected -- log and keep the batch alive
            failed += 1
            logger.exception("  index=%d unexpected error", job.index)
            log_failure(
                args.failures, job.index, job.variant, f"{type(exc).__name__}: {exc}"
            )
            continue

        record = {
            "index": job.index,
            "prompt_variant": job.variant,
            "text": polished,
            "original_text": job.text,
            "model": "dry-run-stub" if args.dry_run else args.model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        append_cache(args.cache, record)
        cache[job.index] = record
        succeeded += 1

    logger.info("Done: %d succeeded, %d failed this run.", succeeded, failed)
    if failed:
        logger.warning(
            "See %s for the failures. Re-run to retry only those (cached "
            "essays are skipped).",
            args.failures,
        )

    written = write_output(cache, args.out)
    if written < len(jobs):
        logger.warning(
            "%d of %d essays are still missing from the output.",
            len(jobs) - written,
            len(jobs),
        )


if __name__ == "__main__":
    main()

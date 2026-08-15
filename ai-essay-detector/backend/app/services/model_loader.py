"""The single place GPT-2 and its tokenizer are loaded.

Everything else imports :func:`get_gpt2` and receives the already-resident
bundle. ``load_gpt2()`` is called exactly once, from the FastAPI lifespan
handler in ``app.main``; it is idempotent and lock-guarded so an accidental
second call returns the same objects rather than pulling ~500MB of weights
into memory again.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_bundle: "GPT2Bundle | None" = None


@dataclass(frozen=True)
class GPT2Bundle:
    """A loaded causal LM plus everything callers need to use it."""

    tokenizer: object
    model: object
    device: torch.device
    max_window_tokens: int
    stride_tokens: int
    name: str

    def encode(self, text: str):
        """Tokenize with char offsets, no special tokens, no truncation."""
        return self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )


def _resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if setting == "cuda" and not torch.cuda.is_available():
        logger.warning("TORCH_DEVICE=cuda requested but no CUDA is available; using CPU.")
        return torch.device("cpu")
    return torch.device(setting)


def load_gpt2(settings: Settings | None = None) -> GPT2Bundle:
    """Load GPT-2 once and cache it for the life of the process."""
    global _bundle

    if _bundle is not None:
        return _bundle

    with _lock:
        # Re-check: another thread may have loaded while we waited.
        if _bundle is not None:
            return _bundle

        settings = settings or get_settings()
        started = time.perf_counter()

        if settings.torch_num_threads > 0:
            torch.set_num_threads(settings.torch_num_threads)

        device = _resolve_device(settings.torch_device)
        cache_dir = str(settings.hf_cache_dir) if settings.hf_cache_dir else None

        logger.info("Loading %s onto %s ...", settings.gpt2_model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(
            settings.gpt2_model_name, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.gpt2_model_name, cache_dir=cache_dir
        )
        model.to(device)
        model.eval()
        # Nothing here ever backpropagates; drop the autograd bookkeeping.
        model.requires_grad_(False)

        # GPT-2 has no pad token; scoring never pads, but downstream helpers
        # are happier when the attribute exists.
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_window = int(
            getattr(model.config, "n_positions", 0)
            or getattr(model.config, "max_position_embeddings", 0)
            or settings.perplexity_window_tokens
        )
        window = min(settings.perplexity_window_tokens, model_window)
        stride = min(settings.perplexity_window_stride, window - 1)

        bundle = GPT2Bundle(
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_window_tokens=window,
            stride_tokens=stride,
            name=settings.gpt2_model_name,
        )

        if settings.warmup_on_startup:
            _warmup(bundle)

        _bundle = bundle
        logger.info(
            "GPT-2 ready in %.2fs (window=%d, stride=%d)",
            time.perf_counter() - started,
            window,
            stride,
        )
        return _bundle


def _warmup(bundle: GPT2Bundle) -> None:
    """One tiny forward pass so the first real request skips lazy init."""
    try:
        with torch.inference_mode():
            ids = torch.tensor(
                [bundle.tokenizer("warm up", add_special_tokens=False)["input_ids"]],
                device=bundle.device,
            )
            if ids.numel():
                bundle.model(ids)
    except Exception:  # pragma: no cover - warmup must never block startup
        logger.warning("GPT-2 warmup pass failed; continuing.", exc_info=True)


def get_gpt2() -> GPT2Bundle:
    """Return the loaded bundle, or raise if startup never completed."""
    if _bundle is None:
        raise RuntimeError(
            "GPT-2 is not loaded. It is loaded once during application startup; "
            "check the boot logs for the failure."
        )
    return _bundle


def is_loaded() -> bool:
    return _bundle is not None


def reset() -> None:
    """Drop the singleton. Intended for tests only."""
    global _bundle
    with _lock:
        _bundle = None

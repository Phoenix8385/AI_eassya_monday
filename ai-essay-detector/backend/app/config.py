"""Application settings.

Every tunable value in the backend originates here and is read from the
environment. No other module hardcodes a path, an origin, a model name or a
threshold -- they import ``get_settings()`` instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved once: <repo>/ai-essay-detector/backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-driven configuration.

    ``protected_namespaces`` is cleared because several fields legitimately
    begin with ``model_`` (MODEL_PATH, MODEL_NAME), which Pydantic otherwise
    reserves for its own attributes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- HTTP -------------------------------------------------------------
    # Comma-separated so a single env var can carry dev + prod origins, e.g.
    # "http://localhost:5173,https://ai-essay-detector.vercel.app"
    cors_origin: str = Field(
        default="http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )
    api_title: str = "AI Essay Detector API"
    api_version: str = "0.1.0"

    # --- Models -----------------------------------------------------------
    model_path: Path = Field(
        default=BACKEND_ROOT / "app" / "data" / "classifier.joblib",
        description="Path to the trained scikit-learn classifier (.joblib).",
    )
    gpt2_model_name: str = Field(
        default="openai-community/gpt2",
        description="HuggingFace repo id for the perplexity language model.",
    )
    hf_cache_dir: Path | None = Field(
        default=None,
        description="Override the HuggingFace cache location (optional).",
    )
    torch_device: str = Field(
        default="cpu",
        description="Torch device for GPT-2: 'cpu', 'cuda', or 'auto'.",
    )
    torch_num_threads: int = Field(
        default=0,
        ge=0,
        description="Torch intra-op threads. 0 leaves the torch default.",
    )
    warmup_on_startup: bool = Field(
        default=True,
        description="Run one tiny forward pass at boot so request #1 is not "
        "paying for torch's lazy kernel initialisation.",
    )

    # --- Analysis ---------------------------------------------------------
    min_essay_chars: int = Field(
        default=200,
        ge=1,
        description="Minimum accepted essay length; also the schema constraint.",
    )
    max_essay_chars: int = Field(
        default=25_000,
        ge=1,
        description="Maximum accepted essay length.",
    )
    perplexity_window_tokens: int = Field(
        default=1024,
        ge=64,
        description="Token window for the GPT-2 forward pass.",
    )
    perplexity_window_stride: int = Field(
        default=512,
        ge=1,
        description="Overlap stride so windowed tokens keep left context.",
    )
    sentence_zscore_offset: float = Field(
        default=1.0,
        ge=0.0,
        description="How many standard deviations more predictable than typical "
        "human writing a sentence must be before its likelihood crosses 0.5. "
        "Without an offset, half of any genuinely human essay sits above the "
        "human mean by definition and gets flagged.",
    )
    sentence_zscore_slope: float = Field(
        default=1.4,
        gt=0.0,
        description="Steepness of the sentence-likelihood sigmoid.",
    )
    sentence_flag_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Per-sentence AI-likelihood above which a sentence is flagged.",
    )
    ai_decision_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Essay-level probability above which the verdict is 'ai'.",
    )
    max_explained_sentences: int = Field(
        default=200,
        ge=1,
        description="Cap on per-sentence records returned in the response.",
    )

    # --- Ops --------------------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("max_essay_chars")
    @classmethod
    def _max_exceeds_min(cls, v: int, info) -> int:
        minimum = info.data.get("min_essay_chars")
        if minimum is not None and v < minimum:
            raise ValueError("MAX_ESSAY_CHARS must be >= MIN_ESSAY_CHARS")
        return v

    @field_validator("perplexity_window_stride")
    @classmethod
    def _stride_fits_window(cls, v: int, info) -> int:
        window = info.data.get("perplexity_window_tokens")
        if window is not None and v >= window:
            raise ValueError(
                "PERPLEXITY_WINDOW_STRIDE must be < PERPLEXITY_WINDOW_TOKENS"
            )
        return v

    @property
    def cors_origins(self) -> list[str]:
        """CORS_ORIGIN split into a list, blanks dropped.

        ``"*"`` is passed through untouched so the wildcard still works.
        """
        return [
            origin.strip()
            for origin in self.cors_origin.split(",")
            if origin.strip()
        ]

    @property
    def resolved_model_path(self) -> Path:
        """MODEL_PATH as an absolute path, relative entries anchored to the
        backend root rather than to whatever cwd uvicorn happened to start in.
        """
        path = self.model_path.expanduser()
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()

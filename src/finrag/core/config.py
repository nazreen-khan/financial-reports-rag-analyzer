"""
finrag.core.config
~~~~~~~~~~~~~~~~~~
Centralised, typed configuration using pydantic-settings.

All settings are read from environment variables (or a .env file).
Usage anywhere in the codebase:

    from finrag.core.config import settings
    print(settings.openai_model)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Enumerations ──────────────────────────────────────────────────────────────

class AppEnv(str, Enum):
    LOCAL = "local"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMBackend(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    WEAVIATE = "weaviate"  # future


# ── Settings Model ────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Single source of truth for all runtime configuration.
    Values are loaded from environment variables, with .env file as fallback.
    Pydantic validates types and raises at startup — fail fast, not at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # ignore unknown env vars (CI safety)
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.LOCAL
    app_log_level: LogLevel = LogLevel.INFO
    app_log_dir: Path = Path("logs")
    app_request_timeout_seconds: int = 60

    # ── LLM Backend ───────────────────────────────────────────────────────────
    llm_backend: LLMBackend = LLMBackend.OPENAI

    # OpenAI
    openai_api_key: str = Field(default="", repr=False)   # repr=False hides from logs
    openai_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o"

    # Local OSS (Ollama / llama.cpp)
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "mistral:7b-instruct"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_cache_dir: Path = Path("data/.embedding_cache")

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_backend: VectorStoreBackend = VectorStoreBackend.CHROMA
    chroma_persist_dir: Path = Path("data/index/chroma")
    chroma_collection_name: str = "finrag_chunks"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 6
    retrieval_hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── EDGAR / Data ──────────────────────────────────────────────────────────
    edgar_user_agent: str = "FinRAG research@example.com"
    edgar_rate_limit_calls: int = 8
    edgar_rate_limit_period: int = 1
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")

    # ── LlamaParse (Optional) ─────────────────────────────────────────────────
    llamaparse_api_key: str = Field(default="", repr=False)
    llamaparse_enabled: bool = False

    # ── Safety / Guardrails ───────────────────────────────────────────────────
    guardrails_enabled: bool = True
    openai_moderation_enabled: bool = False

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── LangSmith (Optional) ──────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = Field(default="", repr=False)
    langchain_project: str = "finrag-local"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("retrieval_hybrid_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("retrieval_hybrid_alpha must be between 0.0 and 1.0")
        return v

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PROD

    @property
    def is_local(self) -> bool:
        return self.app_env == AppEnv.LOCAL

    @property
    def llamaparse_available(self) -> bool:
        return self.llamaparse_enabled and bool(self.llamaparse_api_key)

    @property
    def openai_available(self) -> bool:
        return bool(self.openai_api_key)

    def ensure_dirs(self) -> None:
        """Create all required data directories on first run."""
        dirs = [
            self.app_log_dir,
            self.data_raw_dir,
            self.data_processed_dir,
            self.chroma_persist_dir,
            self.embedding_cache_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ── Singleton accessor ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    The cache means .env is parsed exactly once per process — safe and fast.
    Call get_settings.cache_clear() in tests to reload config.
    """
    return Settings()


# Convenience alias used throughout the codebase:  from finrag.core.config import settings
settings: Settings = get_settings()

"""Configuration and settings.

Windlass reads configuration from four places, later sources winning:

1. Built-in defaults.
2. A config file (``windlass.toml`` / ``.json`` / ``.yaml``), when present.
3. Environment variables — both ``WINDLASS_*`` and the conventional provider
   variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...).
4. Explicit keyword arguments passed to a builder.

Nothing is required to get started: with no configuration at all the
dependency-free ``fake`` LLM, ``hash`` embeddings and ``memory`` vector store
give you a working pipeline for tests and demos.

Example:
    >>> from windlass import configure, settings
    >>> _ = configure(default_llm="fake", request_timeout=30.0)
    >>> settings().default_llm
    'fake'
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from windlass.core.exceptions import ConfigurationError

__all__ = [
    "CacheConfig",
    "RetryConfig",
    "WindlassSettings",
    "configure",
    "load_config_file",
    "reset_settings",
    "settings",
]


class RetryConfig(BaseSettings):
    """Retry policy applied to every provider call.

    Attributes:
        attempts: Total tries including the first. ``1`` disables retrying.
        initial_delay: Seconds before the first retry.
        max_delay: Ceiling for the exponential backoff.
        multiplier: Backoff growth factor.
        jitter: Random fraction (0-1) added to each delay to avoid thundering
            herds when many workers retry at once.

    Example:
        >>> RetryConfig(attempts=5).max_delay
        30.0
    """

    model_config = SettingsConfigDict(env_prefix="WINDLASS_RETRY_", extra="ignore")

    attempts: int = Field(default=3, ge=1, le=10)
    initial_delay: float = Field(default=0.5, ge=0)
    max_delay: float = Field(default=30.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter: float = Field(default=0.2, ge=0.0, le=1.0)


class CacheConfig(BaseSettings):
    """Response caching policy.

    Attributes:
        enabled: Master switch. Off by default — caching a non-deterministic
            model is a decision, not a default.
        backend: ``memory`` (LRU, per process) or ``disk`` (needs the ``cache``
            extra).
        ttl: Seconds an entry stays fresh. ``None`` means forever.
        max_size: Entry ceiling for the in-memory backend.
        directory: Where the disk backend stores its data.
    """

    model_config = SettingsConfigDict(env_prefix="WINDLASS_CACHE_", extra="ignore")

    enabled: bool = False
    backend: Literal["memory", "disk"] = "memory"
    ttl: float | None = 3600.0
    max_size: int = Field(default=1024, ge=1)
    directory: Path = Field(default_factory=lambda: Path.home() / ".cache" / "windlass")


class WindlassSettings(BaseSettings):
    """Process-wide defaults and credentials.

    Every field can be set with an environment variable. Provider keys accept
    both the Windlass-prefixed name and the vendor's conventional name, so an
    existing ``OPENAI_API_KEY`` in your shell just works.

    Attributes:
        default_llm: Provider used when a builder does not name one.
        default_model: Model id used when a builder does not name one.
        default_embedding: Embedding provider used by RAG builders.
        default_vectordb: Vector store used by RAG builders.
        default_chunker: Chunking strategy used by RAG builders.
        default_retriever: Retrieval strategy used by RAG builders.
        temperature: Default sampling temperature.
        max_tokens: Default completion ceiling; ``None`` defers to the provider.
        request_timeout: Per-request timeout in seconds.
        max_concurrency: Ceiling on in-flight provider calls for batch work.
        batch_size: Default batch size for embedding and indexing.
        openai_api_key: Credential for OpenAI-compatible endpoints.
        openai_base_url: Override for proxies and Azure-style gateways.
        anthropic_api_key: Credential for Anthropic.
        google_api_key: Credential for Gemini.
        groq_api_key: Credential for Groq.
        huggingface_api_key: Token for the HuggingFace Inference API.
        ollama_base_url: Where the local Ollama daemon listens.
        pinecone_api_key: Credential for Pinecone.
        langsmith_api_key: Credential for LangSmith tracing.
        langfuse_public_key: Public key for Langfuse.
        langfuse_secret_key: Secret key for Langfuse.
        langfuse_host: Langfuse endpoint.
        telemetry: Whether Windlass may emit anonymous usage counters. Off by
            default; Windlass ships no telemetry backend, the flag exists so
            downstream distributions can honour it.
        strict_plugins: Fail loudly when a third-party plugin cannot load.
        retry: Retry policy.
        cache: Cache policy.

    Example:
        >>> s = WindlassSettings(default_llm="anthropic", temperature=0.0)
        >>> s.default_llm, s.temperature
        ('anthropic', 0.0)
    """

    model_config = SettingsConfigDict(
        env_prefix="WINDLASS_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
        # Credential fields declare validation aliases so the conventional
        # provider env vars work. Without this, those same fields could only be
        # populated by alias — and configure() round-trips settings by field
        # name, which would silently drop every API key.
        populate_by_name=True,
    )

    # -- component defaults ----------------------------------------------
    default_llm: str = "fake"
    default_model: str = ""
    default_embedding: str = "hash"
    default_vectordb: str = "memory"
    default_chunker: str = "recursive"
    default_retriever: str = "vector"

    # -- generation defaults ---------------------------------------------
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    request_timeout: float = Field(default=60.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1, le=256)
    batch_size: int = Field(default=64, ge=1)

    # -- credentials ------------------------------------------------------
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("WINDLASS_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("WINDLASS_OPENAI_BASE_URL", "OPENAI_BASE_URL")
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("WINDLASS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WINDLASS_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"
        ),
    )
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("WINDLASS_GROQ_API_KEY", "GROQ_API_KEY")
    )
    huggingface_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WINDLASS_HUGGINGFACE_API_KEY", "HUGGINGFACE_API_KEY", "HF_TOKEN"
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("WINDLASS_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
    )
    pinecone_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("WINDLASS_PINECONE_API_KEY", "PINECONE_API_KEY")
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WINDLASS_LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"
        ),
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("WINDLASS_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("WINDLASS_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY"),
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        # LANGFUSE_BASE_URL is accepted because the vendor SDK reads it too. Without
        # it, a project on the US region configured that way silently disagrees with
        # Windlass: `windlass config` reports the EU default while the vendor client
        # quietly uses the right host, so the two only agree by luck.
        validation_alias=AliasChoices(
            "WINDLASS_LANGFUSE_HOST", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"
        ),
    )

    # -- behaviour --------------------------------------------------------
    telemetry: bool = False
    strict_plugins: bool = False
    project: str = "windlass"

    retry: RetryConfig = Field(default_factory=RetryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @field_validator("openai_base_url", "ollama_base_url", "langfuse_host", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.rstrip("/") or value
        return value

    # -- helpers ----------------------------------------------------------
    def secret(self, field: str) -> str | None:
        """Return a credential's plaintext value, or ``None`` when unset.

        Args:
            field: Attribute name, e.g. ``"openai_api_key"``.

        Returns:
            The secret value, or ``None``.

        Raises:
            ConfigurationError: If ``field`` is not a known setting.
        """
        if not hasattr(self, field):
            raise ConfigurationError(f"Unknown setting {field!r}.")
        value = getattr(self, field)
        if value is None:
            return None
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)

    def require_secret(self, field: str, *, provider: str, env_var: str) -> str:
        """Return a credential or raise a clear authentication error.

        Args:
            field: Attribute name holding the credential.
            provider: Provider name for the error message.
            env_var: The environment variable a user would normally set.

        Returns:
            The secret value.

        Raises:
            AuthenticationError: When the credential is missing.
        """
        value = self.secret(field)
        if not value:
            from windlass.core.exceptions import AuthenticationError

            raise AuthenticationError(
                f"No API key configured for the {provider} provider.",
                provider=provider,
                hint=(
                    f"Set {env_var} in your environment, add it to a .env file, or pass\n"
                    f"    Windlass.agent().llm('{provider}', api_key='...')"
                ),
            )
        return value

    def masked(self) -> dict[str, Any]:
        """Return settings as a dict with every secret replaced by ``'***'``.

        Safe to log or print. Used by ``windlass config``.
        """
        data = self.model_dump(mode="json")
        for key, value in list(data.items()):
            if isinstance(getattr(self, key, None), SecretStr):
                data[key] = "***" if value else None
        return data


_LOCK = threading.RLock()
_SETTINGS: WindlassSettings | None = None


def settings() -> WindlassSettings:
    """Return the process-wide settings, constructing them on first use.

    Returns:
        The active :class:`WindlassSettings`.

    Example:
        >>> isinstance(settings().request_timeout, float)
        True
    """
    global _SETTINGS
    if _SETTINGS is None:
        with _LOCK:
            if _SETTINGS is None:
                _SETTINGS = _build_settings()
    return _SETTINGS


def _build_settings() -> WindlassSettings:
    """Construct settings from the config file (if any) plus the environment."""
    overrides: dict[str, Any] = {}
    path = _discover_config_file()
    if path is not None:
        overrides = load_config_file(path)
    try:
        return WindlassSettings(**overrides)
    except Exception as exc:
        raise ConfigurationError(
            f"Invalid Windlass configuration: {exc}",
            hint="Check your environment variables and windlass.toml.",
        ) from exc


def _discover_config_file() -> Path | None:
    """Find a config file via ``WINDLASS_CONFIG`` or the conventional names."""
    explicit = os.getenv("WINDLASS_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigurationError(
                f"WINDLASS_CONFIG points at {path}, which does not exist.",
                hint="Unset WINDLASS_CONFIG or create the file.",
            )
        return path
    for name in ("windlass.toml", "windlass.json", "windlass.yaml", "windlass.yml"):
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    return None


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a Windlass config file into a plain dict.

    Supported formats are TOML (stdlib), JSON (stdlib) and YAML (requires
    ``pyyaml``). A ``[windlass]`` / ``windlass:`` top-level table is unwrapped so
    the file can also live inside a shared config.

    Args:
        path: Path to the config file.

    Returns:
        The parsed mapping, ready to pass to :class:`WindlassSettings`.

    Raises:
        ConfigurationError: For a missing file, an unknown extension, a parse
            error, or a non-mapping document.

    Example:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "windlass.json"
        >>> _ = p.write_text('{"windlass": {"default_llm": "fake"}}')
        >>> load_config_file(p)
        {'default_llm': 'fake'}
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise ConfigurationError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            import tomllib

            data = tomllib.loads(path.read_text("utf-8"))
        elif suffix == ".json":
            data = json.loads(path.read_text("utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationError(
                    "YAML config files require PyYAML.",
                    hint="Run: pip install pyyaml — or use windlass.toml instead.",
                ) from exc
            data = yaml.safe_load(path.read_text("utf-8")) or {}
        else:
            raise ConfigurationError(
                f"Unsupported config format {suffix!r}.",
                hint="Use .toml, .json or .yaml.",
            )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Could not parse {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level.")
    if "windlass" in data and isinstance(data["windlass"], dict):
        data = data["windlass"]
    return data


def configure(**overrides: Any) -> WindlassSettings:
    """Update the process-wide settings.

    Values are merged onto the current settings, so partial updates are fine.

    Args:
        **overrides: Any field of :class:`WindlassSettings`. Nested ``retry`` and
            ``cache`` accept either a dict or the model instance.

    Returns:
        The new settings object.

    Raises:
        ConfigurationError: If an override is unknown or fails validation.

    Example:
        >>> configure(temperature=0.0, retry={"attempts": 5}).retry.attempts
        5
        >>> reset_settings()
    """
    global _SETTINGS
    with _LOCK:
        current = settings().model_dump()
        unknown = set(overrides) - set(WindlassSettings.model_fields)
        if unknown:
            raise ConfigurationError(
                f"Unknown setting(s): {', '.join(sorted(unknown))}.",
                hint=f"Valid settings: {', '.join(sorted(WindlassSettings.model_fields))}",
            )
        current.update(overrides)
        try:
            _SETTINGS = WindlassSettings(**current)
        except Exception as exc:
            raise ConfigurationError(f"Invalid configuration override: {exc}") from exc
        return _SETTINGS


def reset_settings() -> None:
    """Discard cached settings so the next :func:`settings` call re-reads the env.

    Primarily a test helper.
    """
    global _SETTINGS
    with _LOCK:
        _SETTINGS = None

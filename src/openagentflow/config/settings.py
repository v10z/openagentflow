"""
Configuration settings for Open Agent Flow.

Uses Pydantic Settings for environment variable and file-based configuration.
Follows the externalized configuration pattern from TwinGraph 1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from openagentflow.core.types import LLMProvider, ReasoningStrategy


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAGENTFLOW_LLM_",
        env_file=".env",
        extra="ignore",
    )

    # Default provider and model
    default_provider: LLMProvider = LLMProvider.ANTHROPIC
    default_model: str = "claude-sonnet-4-20250514"

    # API keys (loaded from environment)
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    azure_openai_api_key: SecretStr | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")

    # AWS for Bedrock
    aws_region: str = "us-east-1"
    aws_profile: str | None = None

    # Ollama for local models
    ollama_base_url: str = "http://localhost:11434"

    # Fallback chain (model IDs to try in order)
    fallback_models: list[str] = Field(default_factory=list)

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

    # Default generation parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    default_timeout_seconds: float = 120.0


class MemorySettings(BaseSettings):
    """Memory backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAGENTFLOW_MEMORY_",
        env_file=".env",
        extra="ignore",
    )

    # Short-term memory
    short_term_strategy: Literal["sliding_window", "summarization"] = "sliding_window"
    short_term_max_tokens: int = 8000

    # Long-term memory (pgvector default)
    long_term_backend: str | None = None  # pgvector, chroma, pinecone
    pgvector_connection: SecretStr | None = None
    chroma_persist_dir: str | None = None
    pinecone_api_key: SecretStr | None = None
    pinecone_environment: str | None = None

    # Episodic memory (graph-based)
    episodic_backend: str = "tinkergraph"  # tinkergraph, neptune


class GuardrailSettings(BaseSettings):
    """Guardrail configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAGENTFLOW_GUARDRAILS_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = True
    content_filter_enabled: bool = True

    # Limits
    max_tokens_per_agent: int = 50000
    max_tokens_per_run: int = 100000
    max_cost_per_agent: float = 10.0
    max_cost_per_run: float = 50.0
    max_iterations_per_agent: int = 25

    # Rate limiting
    rate_limit_per_minute: int = 100

    # Circuit breaker
    circuit_breaker_threshold: int = 5  # Failures before tripping
    circuit_breaker_reset_seconds: float = 60.0


class TracingSettings(BaseSettings):
    """Trace recording configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAGENTFLOW_TRACING_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = True

    # Graph backend
    backend: str = "tinkergraph"  # tinkergraph, neptune
    tinkergraph_endpoint: str = "ws://localhost:8182/gremlin"
    neptune_endpoint: str | None = None

    # Recording options
    record_tool_outputs: bool = True
    record_intermediate_thoughts: bool = True
    record_token_usage: bool = True

    # Retention
    retention_days: int = 30

    # Async recording
    buffer_size: int = 1000
    flush_interval_seconds: float = 5.0


class ServerSettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAGENTFLOW_SERVER_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 1

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Authentication
    auth_enabled: bool = False
    jwt_secret: SecretStr | None = None

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100


class OpenAgentFlowSettings(BaseSettings):
    """
    Main configuration for Open Agent Flow.

    Supports loading from:
    - Environment variables (OPENAGENTFLOW_* prefix)
    - .env file
    - YAML/JSON config files
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENAGENTFLOW_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Project info
    project_name: str = "openagentflow"
    environment: str = "development"
    debug: bool = False

    # Nested configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    # Agent defaults
    default_reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT
    default_max_iterations: int = 10
    default_timeout_seconds: float = 300.0

    # Logging
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    log_file: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> OpenAgentFlowSettings:
        """Load settings from a YAML or JSON file."""
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary (excludes secrets)."""
        return self.model_dump(exclude={"llm": {"anthropic_api_key", "openai_api_key"}})


# Global settings instance (lazy-loaded)
_settings: OpenAgentFlowSettings | None = None

# Direct API key storage (for simple configure() calls)
_api_keys: dict[str, str] = {}


def get_settings() -> OpenAgentFlowSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = OpenAgentFlowSettings()
    return _settings


def configure(
    settings: OpenAgentFlowSettings | None = None,
    *,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    azure_openai_api_key: str | None = None,
    azure_openai_endpoint: str | None = None,
    aws_region: str | None = None,
    aws_profile: str | None = None,
    ollama_base_url: str | None = None,
    default_model: str | None = None,
    debug: bool | None = None,
) -> None:
    """
    Configure Open Agent Flow with API keys and settings.

    Simple usage - just provide your API key:
        import openagentflow
        openagentflow.configure(anthropic_api_key="sk-ant-...")

    Full settings:
        from openagentflow.config import OpenAgentFlowSettings
        openagentflow.configure(settings=OpenAgentFlowSettings(...))

    Args:
        settings: Full settings object (optional)
        anthropic_api_key: Anthropic/Claude API key
        openai_api_key: OpenAI API key
        azure_openai_api_key: Azure OpenAI API key
        azure_openai_endpoint: Azure OpenAI endpoint
        aws_region: AWS region for Bedrock
        aws_profile: AWS profile for Bedrock
        ollama_base_url: Ollama server URL
        default_model: Default model to use
        debug: Enable debug mode
    """
    global _settings, _api_keys

    if settings is not None:
        _settings = settings
        return

    # Store API keys directly for provider access
    if anthropic_api_key:
        _api_keys["anthropic"] = anthropic_api_key
    if openai_api_key:
        _api_keys["openai"] = openai_api_key
    if azure_openai_api_key:
        _api_keys["azure_openai"] = azure_openai_api_key
    if azure_openai_endpoint:
        _api_keys["azure_openai_endpoint"] = azure_openai_endpoint
    if aws_region:
        _api_keys["aws_region"] = aws_region
    if aws_profile:
        _api_keys["aws_profile"] = aws_profile
    if ollama_base_url:
        _api_keys["ollama_base_url"] = ollama_base_url
    if default_model:
        _api_keys["default_model"] = default_model

    # Update settings if they exist
    if _settings is not None and debug is not None:
        _settings.debug = debug


def get_api_key(provider: str) -> str | None:
    """
    Get API key for a provider.

    Checks in order:
    1. Direct configure() calls
    2. Environment variables
    3. Settings file

    Args:
        provider: Provider name (anthropic, openai, azure_openai, etc.)

    Returns:
        API key or None if not found
    """
    import os

    # Check direct configuration first
    if provider in _api_keys:
        return _api_keys[provider]

    # Check environment variables
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "azure_openai_endpoint": "AZURE_OPENAI_ENDPOINT",
        "aws_region": "AWS_REGION",
        "aws_profile": "AWS_PROFILE",
    }

    env_var = env_map.get(provider)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]

    # Check settings
    settings = get_settings()
    if provider == "anthropic" and settings.llm.anthropic_api_key:
        return settings.llm.anthropic_api_key.get_secret_value()
    if provider == "openai" and settings.llm.openai_api_key:
        return settings.llm.openai_api_key.get_secret_value()
    if provider == "azure_openai" and settings.llm.azure_openai_api_key:
        return settings.llm.azure_openai_api_key.get_secret_value()

    return None


def is_configured(provider: str) -> bool:
    """Check if a provider is configured with an API key."""
    return get_api_key(provider) is not None


def load_dotenv(path: str | Path | None = None) -> bool:
    """
    Load environment variables from a .env file.

    Args:
        path: Path to .env file. Defaults to .env in current directory.

    Returns:
        True if file was loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv as _load_dotenv

        if path:
            return _load_dotenv(path)
        return _load_dotenv()
    except ImportError:
        # Try manual parsing if python-dotenv not installed
        env_path = Path(path) if path else Path(".env")
        if not env_path.exists():
            return False

        import os

        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                os.environ[key] = value

        return True

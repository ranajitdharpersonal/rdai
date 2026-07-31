"""Configuration helpers for rdai applications and the bundled CLI."""

from .discovery import (
    API_KEY_PLACEHOLDERS,
    DEFAULT_ENV_FILENAME,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_ENV_KEYS,
    discover_api_keys,
    get_api_key,
    provider_key_statuses,
    read_dotenv,
)
from .loader import (
    DEFAULT_CONFIG_FILENAME,
    ConfigError,
    RdaiConfig,
    load_config,
    save_config,
)

__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_ENV_FILENAME",
    "API_KEY_PLACEHOLDERS",
    "PROVIDER_DISPLAY_NAMES",
    "PROVIDER_ENV_KEYS",
    "ConfigError",
    "RdaiConfig",
    "discover_api_keys",
    "get_api_key",
    "load_config",
    "provider_key_statuses",
    "read_dotenv",
    "save_config",
]

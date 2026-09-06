"""Configuration loading and persistence for rdai.

``rdai.yaml`` only contains routing policy.  Provider secrets are loaded from
``.env`` (without overriding real process environment variables) so callers can
start with no configuration at all and progressively opt in to custom routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml
from dotenv import load_dotenv

from .discovery import DEFAULT_ENV_FILENAME, normalize_provider_name

DEFAULT_CONFIG_FILENAME: Final = "rdai.yaml"
VALID_STRATEGIES: Final[frozenset[str]] = frozenset({"smart", "manual"})


class ConfigError(ValueError):
    """Raised when rdai configuration exists but cannot be used safely."""


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve an explicit config path or the local ``rdai.yaml`` path."""

    candidate = (
        Path(config_path)
        if config_path is not None
        else Path.cwd() / DEFAULT_CONFIG_FILENAME
    )
    return candidate.expanduser().resolve()


@dataclass(frozen=True, slots=True)
class RdaiConfig:
    """The routing policy loaded from ``rdai.yaml``.

    ``providers`` is ordered.  Under manual routing the first provider is the
    preferred one and subsequent providers form the natural fallback chain.
    Under smart routing it remains the available/fallback provider order.
    """

    strategy: str = "smart"
    providers: tuple[str, ...] = ()
    path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        strategy = self.strategy.strip().lower()
        if strategy not in VALID_STRATEGIES:
            supported = ", ".join(sorted(VALID_STRATEGIES))
            raise ConfigError(
                f"Unsupported rdai strategy {self.strategy!r}. Expected one of: {supported}."
            )

        normalized_providers: list[str] = []
        for provider in self.providers:
            if not isinstance(provider, str):
                raise ConfigError("Each provider in rdai.yaml must be a string.")
            try:
                normalized = normalize_provider_name(provider)
            except ValueError as error:
                raise ConfigError(str(error)) from error
            if normalized not in normalized_providers:
                normalized_providers.append(normalized)

        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "providers", tuple(normalized_providers))

    @property
    def provider_order(self) -> tuple[str, ...]:
        """Compatibility-friendly alias that makes ordering explicit."""

        return self.providers

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the stable, human-editable rdai.yaml schema."""

        return {"strategy": self.strategy, "providers": list(self.providers)}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        path: Path | None = None,
    ) -> RdaiConfig:
        """Build a config object from parsed YAML data.

        ``provider_order`` is accepted as a legacy/explicit spelling, while
        saved configuration consistently uses the shorter ``providers`` key.
        """

        if value is None:
            return cls(path=path)
        if not isinstance(value, Mapping):
            raise ConfigError("rdai.yaml must contain a top-level YAML mapping.")

        raw_strategy = value.get("strategy", "smart")
        if not isinstance(raw_strategy, str):
            raise ConfigError("The 'strategy' value in rdai.yaml must be a string.")

        raw_providers = value.get("providers", value.get("provider_order", ()))
        providers = _coerce_providers(raw_providers)
        return cls(strategy=raw_strategy, providers=providers, path=path)


def _coerce_providers(value: Any) -> tuple[str, ...]:
    """Validate provider-order YAML while accepting a convenient CSV spelling."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ConfigError("The 'providers' value in rdai.yaml must be a list of strings.")
    return tuple(value)


def load_config(
    config_path: str | Path | None = None,
    *,
    load_environment: bool = True,
    env_file: str | Path | None = None,
) -> RdaiConfig:
    """Load ``rdai.yaml`` and, by default, add its nearby ``.env`` to the environment.

    Missing config and env files are intentionally valid: they enable the SDK's
    advertised zero-config behaviour.  Existing invalid YAML raises
    :class:`ConfigError` with a message suitable for developers.
    """

    path = resolve_config_path(config_path)
    resolved_env_path = (
        Path(env_file).expanduser().resolve()
        if env_file is not None
        else path.parent / DEFAULT_ENV_FILENAME
    )

    if load_environment and resolved_env_path.is_file():
        # A deployment's real environment must always win over a local .env.
        load_dotenv(dotenv_path=resolved_env_path, override=False)

    if not path.is_file():
        return RdaiConfig()

    try:
        with path.open("r", encoding="utf-8") as config_file:
            parsed = yaml.safe_load(config_file)
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"Could not read rdai config at {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Could not parse rdai config at {path}: {error}") from error

    return RdaiConfig.from_mapping(parsed, path=path)


def save_config(
    config: RdaiConfig,
    config_path: str | Path | None = None,
) -> Path:
    """Persist a routing policy to readable YAML and return its final path."""

    path = resolve_config_path(config_path if config_path is not None else config.path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as config_file:
            yaml.safe_dump(
                config.to_mapping(),
                config_file,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
    except OSError as error:
        raise ConfigError(f"Could not write rdai config at {path}: {error}") from error
    return path

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional


def normalize_model(model: Optional[str]) -> Optional[str]:
    """Normalize a model identifier without changing its meaning."""
    if model is None:
        return None

    if not isinstance(model, str):
        raise TypeError("model must be a string or None.")

    normalized = model.strip()

    return normalized or None


def resolve_model(
    requested_model: Optional[str] = None,
    available_models: Iterable[str] = (),
) -> Optional[str]:
    """Resolve a model without using hardcoded provider defaults.

    Resolution order:

    1. Explicitly requested model.
    2. First valid model discovered from the provider.
    3. None when no valid model exists.

    Hardcoded provider model identifiers must not be supplied here.
    Provider-specific discovery belongs in each provider adapter.
    """
    explicit = normalize_model(requested_model)

    if explicit is not None:
        return explicit

    for candidate in available_models:
        normalized = normalize_model(candidate)

        if normalized is not None:
            return normalized

    return None
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
    fallback_models: Iterable[str] = (),
) -> Optional[str]:
    """Resolve a model using explicit, discovered, then fallback priority.

    Resolution order:
        1. Explicitly requested model.
        2. First valid discovered model.
        3. First valid fallback model.
        4. None when no valid candidate exists.
    """

    explicit = normalize_model(requested_model)

    if explicit is not None:
        return explicit

    for candidates in (available_models, fallback_models):
        for candidate in candidates:
            normalized = normalize_model(candidate)

            if normalized is not None:
                return normalized

    return None
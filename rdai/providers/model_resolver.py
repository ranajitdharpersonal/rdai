from __future__ import annotations

from collections.abc import Iterable
from typing import Optional


def normalize_model(model: Optional[str]) -> Optional[str]:
    if model is None:
        return None

    if not isinstance(model, str):
        raise TypeError("model must be a string or None.")

    model = model.strip()
    return model or None


def resolve_model(
    requested_model: Optional[str],
    available_models: Iterable[str] = (),
    fallback_models: Iterable[str] = (),
) -> Optional[str]:
    explicit = normalize_model(requested_model)
    if explicit is not None:
        return explicit

    for candidates in (available_models, fallback_models):
        for candidate in candidates:
            normalized = normalize_model(candidate)
            if normalized is not None:
                return normalized

    return None
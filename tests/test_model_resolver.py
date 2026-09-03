from rdai.providers.model_resolver import resolve_model


def test_explicit_model_wins():
    assert resolve_model(
        "requested",
        available_models=("discovered",),
        fallback_models=("fallback",),
    ) == "requested"


def test_discovered_model_wins_when_no_explicit_model():
    assert resolve_model(
        None,
        available_models=("discovered",),
        fallback_models=("fallback",),
    ) == "discovered"


def test_fallback_model_used_when_no_discovery():
    assert resolve_model(
        None,
        available_models=(),
        fallback_models=("fallback",),
    ) == "fallback"


def test_none_when_nothing_available():
    assert resolve_model(None) is None


def test_blank_explicit_model_is_ignored():
    assert resolve_model(
        "   ",
        available_models=("discovered",),
    ) == "discovered"
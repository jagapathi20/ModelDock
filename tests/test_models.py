"""The model catalog and its per-version input schemas (C2.1, C2.2)."""

import pytest
from pydantic import ValidationError

from common.models import (
    CATALOG,
    IRIS_V1,
    SENTIMENT_V1,
    ModelSpec,
    UnknownModelError,
    get_spec,
    serving_version,
)


def test_catalog_lookup_and_queue_names():
    assert get_spec("sentiment", 1) is SENTIMENT_V1
    assert serving_version("iris") is IRIS_V1
    # Queue names keep the version, so a registry and canary routing can come later (C1.23).
    assert SENTIMENT_V1.queue == "q:sentiment:1"
    assert all(s.queue == f"q:{name}:{v}" for (name, v), s in CATALOG.items())


def test_unknown_model():
    with pytest.raises(UnknownModelError):
        serving_version("nope")
    with pytest.raises(UnknownModelError):
        get_spec("sentiment", 99)


def test_text_input_accepted_up_to_max_chars():
    assert SENTIMENT_V1.validate_input({"text": "x" * 1000}) == {"text": "x" * 1000}


@pytest.mark.parametrize(
    "bad",
    [
        {"text": "x" * 1001},  # over max_chars
        {"text": ""},  # empty
        {"text": 5},  # not a string (strict: no conversion)
        {},  # missing
        {"text": "ok", "extra": 1},  # unexpected field
        ["text"],  # not an object
    ],
)
def test_text_input_rejected(bad):
    with pytest.raises(ValidationError):
        SENTIMENT_V1.validate_input(bad)


def test_features_accept_ints_and_floats():
    assert IRIS_V1.validate_input({"features": [5, 3.5, 1.4, 0.2]}) == {
        "features": [5.0, 3.5, 1.4, 0.2]
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"features": [1.0, 2.0, 3.0]},  # too few
        {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},  # too many
        {"features": ["5.1", 3.5, 1.4, 0.2]},  # string: strict, not converted
        {"features": [True, 3.5, 1.4, 0.2]},  # bool is not a number here
        {"features": [float("nan"), 3.5, 1.4, 0.2]},  # Python's json accepts NaN tokens
        {"features": [float("inf"), 3.5, 1.4, 0.2]},
    ],
)
def test_features_rejected(bad):
    with pytest.raises(ValidationError):
        IRIS_V1.validate_input(bad)


def test_batching_settings_per_version():
    # C1.13, C1.14, D5; iris doesn't batch (C1.8).
    assert (SENTIMENT_V1.max_batch_size, SENTIMENT_V1.max_wait_ms, SENTIMENT_V1.max_length) == (
        4,
        15.0,
        48,
    )
    assert (IRIS_V1.max_batch_size, IRIS_V1.max_wait_ms) == (1, 0.0)
    assert SENTIMENT_V1.runner_options == {"max_length": 48}
    assert IRIS_V1.runner_options == {}


def test_spec_rejects_bad_batching_settings():
    with pytest.raises(ValueError):
        ModelSpec(
            name="s",
            version=1,
            kind="sklearn_classifier",
            artifact="s",
            n_features=4,
            max_batch_size=0,
        )


def test_spec_requires_its_kinds_limit():
    with pytest.raises(ValueError, match="max_chars"):
        ModelSpec(name="t", version=1, kind="transformer_classifier", artifact="t")
    with pytest.raises(ValueError, match="max_length"):
        ModelSpec(name="t", version=1, kind="transformer_classifier", artifact="t", max_chars=10)
    with pytest.raises(ValueError, match="n_features"):
        ModelSpec(name="s", version=1, kind="sklearn_classifier", artifact="s")

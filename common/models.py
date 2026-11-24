"""The model catalog: which model versions exist, how to load them, what input they accept.

In the first release there is no registry (C1.23), so the catalog lives here, in code. It holds
what changes only together with code or artifacts (the runner kind, the artifact directory, the
input limits). Deployment knobs (Redis URL, timeout, ports) stay in environment variables
(common/config.py).

Both process types read it:

- the **API** validates each request against its version's input schema *before* enqueueing it,
  so invalid input is rejected with 422 and never reaches a batch (C2.1);
- the **worker** finds the runner kind and artifact directory of the version it serves, and
  checks at startup that the artifact agrees with the catalog (the ``n_features`` drift check).

The API never loads a model (D3), so every limit it enforces must be written here: it can't ask
the model. That is why ``n_features`` appears both here and inside ``model.joblib``, and why the
worker refuses to start if the two disagree.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

RunnerKind = Literal["transformer_classifier", "sklearn_classifier"]

# A feature value: a finite number. strict=True rejects "5.1" (string) and true/false
# (booleans) instead of quietly converting them; JSON integers are still accepted.
# allow_inf_nan=False matters because Python's JSON parser accepts NaN and Infinity tokens.
_Feature = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class UnknownModelError(LookupError):
    """No catalog entry for this model name (or version)."""


@dataclass(frozen=True)
class ModelSpec:
    """One model version.

    A version is immutable: everything here is part of what "sentiment v1" means. Changing an
    input limit or the artifact means a new version, not an edit (the same rule C1.13 applies to
    ``max_length`` and D10's cache relies on).
    """

    name: str
    version: int
    kind: RunnerKind
    artifact: str  # directory name under Settings.artifacts_dir
    max_chars: int | None = None  # text models: longest accepted text, in characters (C2.2)
    max_length: int | None = None  # text models: tokens kept; longer inputs truncated (C1.13)
    n_features: int | None = None  # feature models: exact number of features
    # Batching (D5, C1.11): a batch runs when it holds max_batch_size requests, or when
    # max_wait_ms has passed since its first request arrived, whichever comes first.
    max_batch_size: int = 1
    max_wait_ms: float = 0.0

    def __post_init__(self) -> None:
        where = f"{self.name} v{self.version}"
        if self.kind == "transformer_classifier" and not (self.max_chars and self.max_length):
            raise ValueError(f"{where}: text models need max_chars and max_length")
        if self.kind == "sklearn_classifier" and not self.n_features:
            raise ValueError(f"{where}: feature models need n_features")
        if self.max_batch_size < 1 or self.max_wait_ms < 0:
            raise ValueError(f"{where}: max_batch_size >= 1 and max_wait_ms >= 0 required")

    @property
    def runner_options(self) -> dict[str, Any]:
        """Constructor arguments for this version's runner (``build_runner(kind, **options)``)."""
        return {"max_length": self.max_length} if self.kind == "transformer_classifier" else {}

    @property
    def queue(self) -> str:
        """Redis list holding this version's requests (section 6): ``q:{model}:{version}``."""
        return f"q:{self.name}:{self.version}"

    @cached_property
    def input_model(self) -> type[BaseModel]:
        """The pydantic model that one request's input must satisfy.

        Built from this version's limits, so ``max_chars`` and ``n_features`` are enforced by the
        schema itself and the error messages name the field. ``extra="forbid"``: an unexpected
        field is a client bug worth reporting, not something to ignore silently.
        """
        config = ConfigDict(extra="forbid")
        name = f"{self.name.capitalize()}V{self.version}Input"
        if self.kind == "transformer_classifier":
            text = Annotated[str, Field(strict=True, min_length=1, max_length=self.max_chars)]
            return create_model(name, __config__=config, text=(text, ...))
        features = Annotated[
            list[_Feature], Field(min_length=self.n_features, max_length=self.n_features)
        ]
        return create_model(name, __config__=config, features=(features, ...))

    def validate_input(self, raw: Any) -> dict[str, Any]:
        """Return the input as a plain dict, or raise pydantic.ValidationError."""
        return self.input_model.model_validate(raw).model_dump()


# --- The catalog -------------------------------------------------------------------------------

SENTIMENT_V1 = ModelSpec(
    name="sentiment",
    version=1,
    kind="transformer_classifier",
    artifact="sentiment-v1",
    # About 4x a 48-token English text. The token limit (C1.13) bounds compute and is applied by
    # truncation in the worker; this one bounds the Redis message and tokenizer work (C2.2).
    max_chars=1000,
    max_length=48,  # C1.13: the longest length whose batches still fit the 150 ms budget
    max_batch_size=4,  # C1.14 (P19: 4 or 8, decided by the load test)
    max_wait_ms=15.0,  # D5
)

IRIS_V1 = ModelSpec(
    name="iris",
    version=1,
    kind="sklearn_classifier",
    artifact="iris-v1",
    n_features=4,
    # Batching off (C1.8): waiting to fill a batch costs far more than the 0.12 ms call. P16
    # proposes trying max_wait_ms = 0 with a larger batch size in the load test.
    max_batch_size=1,
    max_wait_ms=0.0,
)

CATALOG: dict[tuple[str, int], ModelSpec] = {
    (s.name, s.version): s for s in (SENTIMENT_V1, IRIS_V1)
}


def get_spec(name: str, version: int) -> ModelSpec:
    try:
        return CATALOG[(name, version)]
    except KeyError:
        raise UnknownModelError(f"unknown model {name!r} version {version}") from None


def serving_version(name: str) -> ModelSpec:
    """The version the API sends a model's traffic to.

    Each model has exactly one version in the first release, so this is that version. Canary
    routing (future work, C1.22) would replace this with a weighted, sticky choice.
    """
    versions = [s for (n, _), s in CATALOG.items() if n == name]
    if not versions:
        raise UnknownModelError(f"unknown model {name!r}")
    return max(versions, key=lambda s: s.version)

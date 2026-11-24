"""Model runners: one class per kind of model, all with the same interface.

Runner modules import heavy libraries (torch, transformers, sklearn), so this
package doesn't import them eagerly. ``build_runner`` imports only the one
that's asked for, which keeps ``import worker.runners`` cheap.
"""

from worker.runners.base import InvalidInputError, JsonDict, ModelNotLoadedError, ModelRunner

RUNNER_KINDS = ("transformer_classifier", "sklearn_classifier")


def build_runner(kind: str, **options) -> ModelRunner:
    """Create an (unloaded) runner by kind name; ``options`` go to its constructor.

    The worker passes the catalog's per-version settings, e.g. ``max_length=48`` for a
    transformer (C1.13).
    """
    if kind == "transformer_classifier":
        from worker.runners.transformer import TransformerClassifierRunner

        return TransformerClassifierRunner(**options)
    if kind == "sklearn_classifier":
        from worker.runners.tabular import SklearnClassifierRunner

        return SklearnClassifierRunner(**options)
    raise ValueError(f"unknown runner kind {kind!r}; expected one of {RUNNER_KINDS}")


__all__ = [
    "InvalidInputError",
    "JsonDict",
    "ModelNotLoadedError",
    "ModelRunner",
    "RUNNER_KINDS",
    "build_runner",
]

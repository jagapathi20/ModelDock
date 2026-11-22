"""The interface every model runner implements.

A *runner* wraps one kind of model (a Hugging Face transformer, a scikit-learn
pipeline, ...) behind the same two calls, so the worker's batching loop
(Stage 3) never needs to know what kind of model it's driving:

    runner.load(model_dir)              # once, at worker startup
    runner.predict_batch(inputs)        # many times, one call per batch

Inputs and outputs are plain JSON-compatible dicts, matching the wire format
of request and reply messages (docs/DESIGN.md, section 6): an input is the
``"input"`` field of a request message, an output becomes the ``"output"``
field of the reply.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


class ModelNotLoadedError(RuntimeError):
    """predict_batch() was called before load()."""


class ModelRunner(ABC):
    """Base class for all runners.

    Contract:
    - ``load()`` reads from a **local directory only**. It never downloads
      anything. Fetching weights (from Hugging Face now, from object storage
      if the model registry is built; future work, C1.23) is a separate step,
      so the hot path has no network surprises
      and the same code serves every storage backend.
    - ``predict_batch(inputs)`` returns exactly one output per input, **in the
      same order**. The worker matches outputs back to requests by position.
    - Inference is deterministic: models run in eval mode, with no sampling.
      The response cache (D10) relies on this.
    """

    #: Short identifier used in logs, metrics and the benchmark output.
    kind: str = "base"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, model_dir: Path) -> None:
        """Load the model from ``model_dir`` and put it in inference mode."""

    @abstractmethod
    def predict_batch(self, inputs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        """Run one batch. ``len(result) == len(inputs)``, order preserved."""

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotLoadedError(f"{type(self).__name__}.load() has not been called")

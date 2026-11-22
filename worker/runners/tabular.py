"""Runner for scikit-learn classifiers on numeric features (the contrast model).

One call costs about 0.12 ms whether it holds 1 item or 32, so batching
does cut the per-item cost. It stays off (``max_batch_size = 1``) because
waiting up to ``max_wait`` (15 ms) to fill a batch costs far more than the
fraction of a millisecond it would save (C1.8; P16 proposes trying
``max_wait_ms = 0`` with larger batches instead). The model exists to show the
runner interface handles more than one kind of model.

Artifact layout (written by ``scripts/train_iris.py``)::

    <model_dir>/model.joblib    a dict: {"pipeline", "labels", "n_features"}
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from worker.runners.base import InvalidInputError, JsonDict, ModelRunner

ARTIFACT_FILENAME = "model.joblib"


class SklearnClassifierRunner(ModelRunner):
    """Classification with any fitted scikit-learn estimator that has predict_proba.

    Input:  ``{"features": [5.1, 3.5, 1.4, 0.2]}``
    Output: ``{"label": "setosa", "score": 0.97}``
    """

    kind = "sklearn_classifier"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Any = None
        self._labels: list[str] = []
        self.n_features = 0

    def load(self, model_dir: Path) -> None:
        # joblib.load is pickle underneath, and unpickling can run arbitrary
        # code. Only load artifacts we produced ourselves. The model registry
        # (future work, C1.23) would add a second guard: the file's SHA-256
        # must match the registry entry (D8).
        bundle = joblib.load(Path(model_dir) / ARTIFACT_FILENAME)
        self._pipeline = bundle["pipeline"]
        self._labels = list(bundle["labels"])
        self.n_features = int(bundle["n_features"])
        self._loaded = True

    def predict_batch(self, inputs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        self._require_loaded()
        if not inputs:
            return []
        x = self._to_matrix(inputs)
        probs = self._pipeline.predict_proba(x)
        best = probs.argmax(axis=1)
        return [
            {"label": self._labels[int(c)], "score": float(probs[row, c])}
            for row, c in enumerate(best)
        ]

    def _to_matrix(self, inputs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Check each input, then build the (n, n_features) matrix.

        Each input is checked on its own so the error names the bad one (``index``). A whole-batch
        ``np.asarray`` would raise a generic ValueError ("inhomogeneous shape") that says neither
        which input was wrong nor whose fault it was.
        """
        rows = []
        for i, item in enumerate(inputs):
            features = item.get("features") if isinstance(item, Mapping) else None
            if not isinstance(features, list | tuple) or len(features) != self.n_features:
                raise InvalidInputError(
                    f"input {i}: expected {self.n_features} features per input", index=i
                )
            try:
                row = np.asarray(features, dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise InvalidInputError(f"input {i}: features must be numbers", index=i) from e
            if not np.isfinite(row).all():
                raise InvalidInputError(f"input {i}: features must be finite", index=i)
            rows.append(row)
        return np.vstack(rows)

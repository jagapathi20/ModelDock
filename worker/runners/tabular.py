"""Runner for scikit-learn classifiers on numeric features (the contrast model).

Inference here costs microseconds to a millisecond, so there's nothing for
batching to save; this model runs with ``max_batch_size = 1`` (D5). It exists
to show the runner interface (and later the registry) handles more than one
kind of model.

Artifact layout (written by ``scripts/train_iris.py``)::

    <model_dir>/model.joblib    a dict: {"pipeline", "labels", "n_features"}
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from worker.runners.base import JsonDict, ModelRunner

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
        # code. Only load artifacts we produced ourselves. Stage 5 adds the
        # second guard: the file's SHA-256 must match the registry entry.
        bundle = joblib.load(Path(model_dir) / ARTIFACT_FILENAME)
        self._pipeline = bundle["pipeline"]
        self._labels = list(bundle["labels"])
        self.n_features = int(bundle["n_features"])
        self._loaded = True

    def predict_batch(self, inputs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        self._require_loaded()
        if not inputs:
            return []
        x = np.asarray([item["features"] for item in inputs], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features per input, got shape {x.shape}")
        probs = self._pipeline.predict_proba(x)
        best = probs.argmax(axis=1)
        return [
            {"label": self._labels[int(c)], "score": float(probs[row, c])}
            for row, c in enumerate(best)
        ]

"""Train the contrast model: a scikit-learn classifier on the iris dataset.

    uv run python -m scripts.train_iris --out artifacts/iris-v1

Iris ships inside scikit-learn, so this needs no download. The result is the
artifact layout SklearnClassifierRunner loads: ``<out>/model.joblib``.
"""

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from worker.runners.tabular import ARTIFACT_FILENAME


def train(out_dir: Path, seed: int = 0) -> dict[str, float]:
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target
    )
    # The scaler is part of the pipeline, so the saved artifact contains its
    # fitted mean/std and callers send raw features.
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(x_train, y_train)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "labels": [str(n) for n in data.target_names],
            "n_features": int(data.data.shape[1]),
        },
        out_dir / ARTIFACT_FILENAME,
    )
    return {
        "train_accuracy": float(pipeline.score(x_train, y_train)),
        "test_accuracy": float(pipeline.score(x_test, y_test)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("artifacts/iris-v1"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    metrics = train(args.out, args.seed)
    print(f"saved to {args.out / ARTIFACT_FILENAME}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()

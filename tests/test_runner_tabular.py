import pytest

pytest.importorskip("sklearn")

from scripts.train_iris import train  # noqa: E402
from worker.runners import ModelNotLoadedError, build_runner  # noqa: E402
from worker.runners.tabular import SklearnClassifierRunner  # noqa: E402

SETOSA = {"features": [5.1, 3.5, 1.4, 0.2]}  # first row of the iris dataset
VIRGINICA = {"features": [6.7, 3.0, 5.2, 2.3]}  # a typical virginica


@pytest.fixture(scope="session")
def iris_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("iris-v1")
    metrics = train(d)
    assert metrics["test_accuracy"] > 0.85
    return d


@pytest.fixture
def runner(iris_dir):
    r = SklearnClassifierRunner()
    r.load(iris_dir)
    return r


def test_predicts_known_examples(runner):
    out = runner.predict_batch([SETOSA, VIRGINICA])
    assert [o["label"] for o in out] == ["setosa", "virginica"]
    assert all(0 < o["score"] <= 1 for o in out)


def test_output_order_matches_input_order(runner):
    out = runner.predict_batch([VIRGINICA, SETOSA, VIRGINICA])
    assert [o["label"] for o in out] == ["virginica", "setosa", "virginica"]


def test_batched_result_matches_one_at_a_time(runner):
    """Labels must be identical; scores may differ in the last float digits.

    A batch of rows and a single row can go through different BLAS code paths
    (e.g. Apple Accelerate on macOS vs OpenBLAS on Linux), and floating-point
    addition isn't associative, so exact equality is platform-dependent.
    """
    batch = [SETOSA, VIRGINICA]
    batched = runner.predict_batch(batch)
    singles = [runner.predict_batch([x])[0] for x in batch]
    for b, s in zip(batched, singles, strict=True):
        assert b["label"] == s["label"]
        assert b["score"] == pytest.approx(s["score"], abs=1e-9)


def test_wrong_feature_count_raises(runner):
    with pytest.raises(ValueError, match="expected 4 features"):
        runner.predict_batch([{"features": [1.0, 2.0]}])


def test_empty_batch(runner):
    assert runner.predict_batch([]) == []


def test_predict_before_load_raises():
    with pytest.raises(ModelNotLoadedError):
        SklearnClassifierRunner().predict_batch([SETOSA])


def test_build_runner_returns_sklearn_runner():
    assert isinstance(build_runner("sklearn_classifier"), SklearnClassifierRunner)

"""Tests for TransformerClassifierRunner.

They use a tiny randomly initialized DistilBERT built on the fly (a few
thousand parameters, 20-word vocabulary), so they need no download and run in
about a second. Its predictions are meaningless; what's tested is the
runner's behavior: shape, order, determinism, padding, truncation.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from worker.runners import ModelNotLoadedError, build_runner  # noqa: E402
from worker.runners.transformer import (  # noqa: E402
    TransformerClassifierRunner,
    configure_cpu_threads,
)

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + (
    "the movie was good bad great awful plot fun slow and not very".split()
)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-distilbert")
    (d / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    tokenizer = transformers.DistilBertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    config = transformers.DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    torch.manual_seed(0)
    model = transformers.DistilBertForSequenceClassification(config)
    tokenizer.save_pretrained(d)
    model.save_pretrained(d)
    return d


@pytest.fixture
def runner(tiny_model_dir):
    r = TransformerClassifierRunner(max_length=32)
    r.load(tiny_model_dir)
    return r


TEXTS = [
    {"text": "the movie was good"},
    {"text": "bad"},
    {"text": "the plot was not very fun and the movie was slow and awful"},
]


def test_one_output_per_input_with_valid_labels(runner):
    out = runner.predict_batch(TEXTS)
    assert len(out) == len(TEXTS)
    for o in out:
        assert set(o) == {"label", "score"}
        assert o["label"] in {"NEGATIVE", "POSITIVE"}
        # With two classes the winning probability is always >= 0.5.
        assert 0.5 <= o["score"] <= 1.0


def test_empty_batch_returns_empty_list(runner):
    assert runner.predict_batch([]) == []


def test_predict_before_load_raises():
    with pytest.raises(ModelNotLoadedError):
        TransformerClassifierRunner().predict_batch(TEXTS)


def test_model_is_in_eval_mode(runner):
    # Training mode would enable dropout and make outputs random.
    assert runner._model.training is False


def test_same_input_gives_same_output(runner):
    assert runner.predict_batch(TEXTS) == runner.predict_batch(TEXTS)


def test_batched_result_matches_one_at_a_time(runner):
    """Padding must not change the answer.

    The texts have different lengths, so in a batch the short ones are padded.
    The attention mask makes the model ignore padding; if it didn't, a text's
    score would depend on which other texts shared its batch.

    Scores can still differ in the last float digits, because a different
    batch shape can take a different (but equally valid) path through the
    matrix-multiply kernels. Labels must be identical.
    """
    batched = runner.predict_batch(TEXTS)
    singles = [runner.predict_batch([t])[0] for t in TEXTS]
    for b, s in zip(batched, singles, strict=True):
        assert b["label"] == s["label"]
        assert b["score"] == pytest.approx(s["score"], abs=1e-5)


def test_long_inputs_are_truncated_to_max_length(runner):
    long_text = {"text": " ".join(["movie"] * 500)}
    encoded = runner.preprocess([long_text, {"text": "good"}])
    assert encoded["input_ids"].shape == (2, 32)


def test_batch_is_padded_to_its_longest_input(runner):
    encoded = runner.preprocess([{"text": "good"}, {"text": "the movie was good"}])
    # [CLS] the movie was good [SEP] = 6 tokens; "good" is padded up to 6.
    assert encoded["input_ids"].shape == (2, 6)
    assert encoded["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0]


def test_build_runner_returns_transformer_runner():
    assert isinstance(build_runner("transformer_classifier"), TransformerClassifierRunner)


def test_build_runner_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown runner kind"):
        build_runner("onnx")


def test_configure_cpu_threads():
    before = torch.get_num_threads()
    try:
        configure_cpu_threads(1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
    with pytest.raises(ValueError):
        configure_cpu_threads(0)

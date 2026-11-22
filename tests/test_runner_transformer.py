"""Tests for TransformerClassifierRunner.

They use a tiny randomly initialized DistilBERT built on the fly (a few
thousand parameters, 20-word vocabulary), so they need no download and run in
about a second. Its predictions are meaningless; what's tested is the
runner's behavior: shape, order, determinism, padding, truncation.
"""

from types import SimpleNamespace

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


# Different token counts on purpose: [CLS] + words + [SEP] = 3, 4, 6 and 7 tokens.
ORDER_TEXTS = [
    {"text": "good"},
    {"text": "great plot"},
    {"text": "the movie was awful"},
    {"text": "not very fun and slow"},
]


class _LengthModel:
    """Stand-in for the network: logits ``[0, n]``, where n is the row's number of real tokens.

    So each input's score is known in advance (softmax of ``[0, n]`` picks label 1 with
    probability sigmoid(n)) and differs between inputs of different lengths.
    """

    def __call__(self, input_ids, attention_mask, **_):
        n = attention_mask.sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([torch.zeros_like(n), n], dim=1))


def test_output_order_matches_input_order(runner, monkeypatch):
    """Output i belongs to input i (base.py's contract; the worker relies on it).

    The worker matches outputs to requests, and so to request_ids, by position. If the runner
    reordered them, every client in the batch would get someone else's answer, and nothing else
    would fail.

    The tiny random model can't be used to tell inputs apart: its scores for these texts are all
    about 0.5075, within 3e-6 of each other. So the network is replaced by _LengthModel, whose
    output for each input is known exactly. The runner's own code (tokenizer, padding,
    forward(), postprocess()) still runs: a reordering bug there is what this test catches. A
    real network can't reorder rows; runner code (e.g. sorting by length, future token-budget
    batching) can.
    """
    monkeypatch.setattr(runner, "_model", _LengthModel())

    def expected_score(item):
        n = len(runner._tokenizer(item["text"])["input_ids"])
        return torch.sigmoid(torch.tensor(float(n))).item()

    expected = [expected_score(t) for t in ORDER_TEXTS]
    # Guard: distinct lengths give distinct expected scores, so any swap is visible.
    assert len({round(e, 6) for e in expected}) == len(ORDER_TEXTS)

    perm = [2, 0, 3, 1]
    for order in (range(len(ORDER_TEXTS)), perm):
        batch = [ORDER_TEXTS[i] for i in order]
        outputs = runner.predict_batch(batch)
        assert [o["score"] for o in outputs] == pytest.approx(
            [expected[i] for i in order], abs=1e-6
        )
        assert all(o["label"] == "POSITIVE" for o in outputs)  # label 1 always wins: n > 0


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

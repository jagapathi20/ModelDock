"""The sentiment model download is pinned to one Hub commit.

No network: from_pretrained is replaced by a fake that records the revision it was asked for.
"""

import re

import pytest

pytest.importorskip("transformers")

from scripts import fetch_sentiment_model as fetch_mod  # noqa: E402


def test_default_revision_is_a_full_commit_hash():
    # A branch name ("main") moves; a short hash isn't accepted everywhere.
    assert re.fullmatch(r"[0-9a-f]{40}", fetch_mod.DEFAULT_REVISION)


class _Fake:
    calls: list[tuple[str, str | None]] = []

    @classmethod
    def from_pretrained(cls, model_id, revision=None):
        cls.calls.append((model_id, revision))
        return cls()

    def save_pretrained(self, out_dir):
        pass


@pytest.mark.parametrize("revision", [None, "0" * 40])
def test_fetch_passes_the_revision_to_the_hub(monkeypatch, tmp_path, revision):
    _Fake.calls = []
    monkeypatch.setattr(fetch_mod, "AutoTokenizer", _Fake)
    monkeypatch.setattr(fetch_mod, "AutoModelForSequenceClassification", _Fake)

    kwargs = {} if revision is None else {"revision": revision}
    fetch_mod.fetch("some/model", tmp_path / "out", **kwargs)

    expected = revision or fetch_mod.DEFAULT_REVISION  # no argument means the pinned commit
    assert _Fake.calls == [("some/model", expected), ("some/model", expected)]

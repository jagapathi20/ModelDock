"""Every benchmark result file the docs cite must exist in the repo and be a real result.

DESIGN.md's rule 3: "Every entry names its evidence. Performance figures only come from recorded
runs." The evidence is the JSON that the benchmark scripts write to bench/results/. This test makes
the rule checkable: if a doc cites a file that was never committed (or was renamed or deleted),
the suite fails instead of the claim silently losing its source.

No model or extra dependency needed; it reads text and JSON only.
"""

import json
import re
from pathlib import Path

import pytest

# conftest.py chdirs every test into an empty temp dir, so paths must not be relative to the
# working directory. This file is <repo>/tests/test_evidence.py, so parents[1] is the repo root.
REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "bench" / "results"

# A citation looks like `bench/results/stage1-transformer_classifier-20261110T105742Z.json`.
# Templates such as `bench/results/stage1-<kind>-<timestamp>.json` don't match, because `<`
# is outside the character class.
CITATION = re.compile(r"bench/results/[A-Za-z0-9_.-]+\.json")

# Top-level keys every benchmark script writes (scripts/bench_batching*.py). "machine" is the
# hardware block that makes a number meaningful; "results" holds the measurements.
REQUIRED_KEYS = {"args", "machine", "results"}


def _docs() -> list[Path]:
    return [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]


def _cited() -> dict[str, list[str]]:
    """Map each cited result file to the docs that cite it."""
    cited: dict[str, list[str]] = {}
    for doc in _docs():
        for match in CITATION.findall(doc.read_text(encoding="utf-8")):
            cited.setdefault(match, []).append(str(doc.relative_to(REPO)))
    return cited


def test_docs_cite_some_results():
    # Guards the test itself: if the regex or the doc paths broke, every other check here
    # would pass vacuously.
    assert _cited(), "no bench/results/*.json citations found in README.md or docs/"


def test_every_cited_result_file_exists():
    missing = {path: docs for path, docs in sorted(_cited().items()) if not (REPO / path).is_file()}
    assert not missing, "cited but not in the repo:\n" + "\n".join(
        f"  {path}  (cited in {', '.join(sorted(set(docs)))})" for path, docs in missing.items()
    )


@pytest.mark.parametrize("path", sorted(RESULTS_DIR.glob("*.json")), ids=lambda p: p.name)
def test_result_file_is_a_benchmark_result(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert REQUIRED_KEYS <= data.keys(), f"missing keys: {REQUIRED_KEYS - data.keys()}"
    assert data["results"], "no measurements"

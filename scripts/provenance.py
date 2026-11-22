"""Record which code produced a benchmark result.

Every result file in bench/results/ is evidence for a claim in docs/DESIGN.md (rule 3). The JSON
already records the arguments and the machine; this adds the code: the script that ran and the git
commit it ran from. With the commit, ``git checkout <commit>`` reproduces the exact code behind a
number, even after the script has changed.

``dirty`` is the important companion field. A commit hash taken from a working tree with uncommitted
edits names code that is *not* what ran, so the result says so instead of pretending to be
traceable. Untracked files are ignored (``--untracked-files=no``): result files from earlier runs
are untracked until committed, and counting them would mark nearly every run dirty.
"""

import subprocess
from pathlib import Path
from typing import Any

# scripts/provenance.py -> the repo root. Not the working directory: the benchmark may be started
# from anywhere, and the tests chdir into a temp dir.
REPO_ROOT = Path(__file__).resolve().parents[1]


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any] | None:
    """``{"commit": <40-hex sha>, "dirty": bool}``, or None if git or the repo isn't available.

    None rather than an exception: a missing git binary (e.g. a slim container) must not stop a
    benchmark; the result just can't be traced to a commit, and the JSON shows that as null.
    """
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(status)}


def provenance(script: str, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The ``"provenance"`` block written into a result file."""
    return {"script": script, "git": git_state(repo)}


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, timeout=5, check=True
    )
    return out.stdout.strip()

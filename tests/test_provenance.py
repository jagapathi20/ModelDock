"""Benchmark results record the script and git commit that produced them."""

import shutil
import subprocess

import pytest

from scripts.provenance import git_state, provenance

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def _head(repo):
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def test_clean_tree_records_the_commit(repo):
    assert git_state(repo) == {"commit": _head(repo), "dirty": False}


def test_uncommitted_change_is_marked_dirty(repo):
    (repo / "a.txt").write_text("two\n")
    assert git_state(repo)["dirty"] is True


def test_untracked_files_do_not_count_as_dirty(repo):
    # New result files are untracked until committed; they must not mark every run dirty.
    (repo / "stage2-result.json").write_text("{}")
    assert git_state(repo)["dirty"] is False


def test_outside_a_repository_returns_none(tmp_path):
    assert git_state(tmp_path) is None


def test_provenance_block_names_the_script(repo):
    assert provenance("bench_batching.py", repo) == {
        "script": "bench_batching.py",
        "git": {"commit": _head(repo), "dirty": False},
    }

"""Native thread caps (C3.2, P15). Each check runs in a fresh interpreter: the caps only take
effect if set before numpy is first imported, and this test process imported it long ago."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from worker.threads import THREAD_ENV_VARS, cap_native_threads

REPO = Path(__file__).resolve().parents[1]


def _python(code: str, extra_env: dict[str, str] | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV_VARS}
    env |= extra_env or {}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_cap_sets_each_variable_but_keeps_explicit_ones(monkeypatch):
    for var in THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")  # set by hand: must win
    values = cap_native_threads(2)
    assert values["OMP_NUM_THREADS"] == "7"
    assert all(values[v] == "2" for v in THREAD_ENV_VARS if v != "OMP_NUM_THREADS")


def test_cap_rejects_zero():
    with pytest.raises(ValueError):
        cap_native_threads(0)


def test_worker_entry_point_imports_no_numpy_or_torch():
    # If it did, the caps set in main() would come too late to affect those libraries.
    loaded = _python(
        "import sys, worker.__main__; "
        "print(sorted(m for m in ('numpy', 'torch', 'sklearn') if m in sys.modules))"
    )
    assert loaded == "[]"


def test_caps_limit_the_pools_numpy_loads():
    pytest.importorskip("numpy")
    pytest.importorskip("threadpoolctl")
    counts = _python(
        "from worker.threads import cap_native_threads; cap_native_threads(2); "
        "import numpy, threadpoolctl; "
        "print([p['num_threads'] for p in threadpoolctl.threadpool_info()])"
    )
    # threadpoolctl doesn't see every library (Apple Accelerate isn't reported), so an empty list
    # is possible on macOS; whatever it does see must respect the cap.
    assert all(n <= 2 for n in eval(counts))

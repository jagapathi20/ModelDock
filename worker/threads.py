"""Cap every native thread pool in a worker process (D7, C1.5; resolves P15, see C3.2).

A worker's threads must not exceed its share of the performance cores. ``torch.set_num_threads``
caps only PyTorch's own pool. numpy (and so scikit-learn) calls a BLAS library with a pool of its
own: Apple's Accelerate on macOS, OpenBLAS on Linux, sometimes MKL; OpenMP-based code has another.
Those read their limits from environment variables **once, when the library is first loaded**,
so the variables must be set before anything imports numpy. Setting them later does nothing.

Hence this module imports nothing heavy, and the worker calls ``cap_native_threads`` before it
builds a runner (runner modules import numpy and torch).
"""

import logging
import os

log = logging.getLogger("modeldock.worker.threads")

# One variable per library that might own a pool in this process.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",  # OpenMP: PyTorch on Linux, scikit-learn's compiled code
    "OPENBLAS_NUM_THREADS",  # OpenBLAS: numpy's BLAS on Linux
    "MKL_NUM_THREADS",  # Intel MKL, if numpy or torch was built with it
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate (vecLib): numpy's BLAS on macOS
)


def cap_native_threads(n: int) -> dict[str, str]:
    """Set each thread-count variable to ``n`` unless it's already set; return the values used.

    ``setdefault``: a value set explicitly in the environment wins, so an experiment can still
    override the cap from the shell.
    """
    if n < 1:
        raise ValueError("thread count must be >= 1")
    return {var: os.environ.setdefault(var, str(n)) for var in THREAD_ENV_VARS}


def log_thread_pools() -> None:
    """Log the native thread pools actually loaded, as threadpoolctl reports them.

    threadpoolctl (a scikit-learn dependency) inspects the libraries loaded in this process. It
    may not see every pool (Apple Accelerate isn't reported, for instance), so this is a check,
    not a guarantee.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        log.info("threadpoolctl not installed; native thread pools not reported")
        return
    for pool in threadpool_info():
        log.info(
            "thread pool: %s (%s) num_threads=%s",
            pool.get("internal_api"),
            pool.get("prefix"),
            pool.get("num_threads"),
        )

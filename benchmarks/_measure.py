"""Wall-clock timing, peak device memory, and the statistics every timing is reported with.

Timings bracket the work with a full device synchronise. A stream or event sync is not enough:
the solver runs on its own stream and joins it back with an event.

The CG iteration is captured into a CUDA graph on its second iteration, keyed on the
preconditioner and the block width, so the first calls after changing either are not
representative. ``timed`` always warms up first.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def sync() -> None:
    import cupy as cp

    cp.cuda.Device().synchronize()


def once[T](fn: Callable[[], T]) -> tuple[float, T]:
    """Milliseconds for one call of ``fn``, and what it returned."""
    sync()
    t0 = time.perf_counter()
    value = fn()
    sync()
    return (time.perf_counter() - t0) * 1e3, value


def timed(fn: Callable[[], object], n: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    return [once(fn)[0] for _ in range(n)]


def stats(samples: list[float] | np.ndarray, *, seed: int = 0) -> dict[str, float]:
    """Median with a 95% bootstrap interval, and the spread of the samples."""
    x = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(x, size=(2000, x.size)), axis=1)
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "median": float(median),
        "ci95": [float(v) for v in np.percentile(medians, [2.5, 97.5])],
        "q1": float(q1),
        "q3": float(q3),
        "iqr_over_median": float((q3 - q1) / median) if median else float("nan"),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def free_pools() -> None:
    """Return cached blocks to the device. Callers must drop their references first."""
    import cupy as cp

    sync()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def used_mib() -> float:
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 2**20


class PeakMemory:
    """Peak device memory in use while the block runs, above what was in use on entry.

    Polls the driver rather than CuPy's pool, so allocations the extension makes itself count.
    """

    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0.0

    def __enter__(self) -> Self:
        import cupy as cp

        self._device = cp.cuda.Device().id
        self._baseline = used_mib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        import cupy as cp

        with cp.cuda.Device(self._device):
            while True:
                self.peak_mib = max(self.peak_mib, used_mib() - self._baseline)
                if self._stop.wait(self.interval_s):
                    return

    def __exit__(self, *exc) -> None:
        sync()
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, used_mib() - self._baseline)

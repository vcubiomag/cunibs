"""Child for test_gil.py: how long a Python thread stalls while the main thread solves."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cunibs import Subject
from cunibs.mesh import load_mesh


def main() -> None:
    subject = Subject(load_mesh(Path(sys.argv[1])))
    solver = subject.context.solver
    n = int(solver.idx.shape[0])
    b = cp.asarray(np.random.default_rng(0).standard_normal((n, 8)))
    x = cp.empty_like(b)
    stream = cp.cuda.get_current_stream().ptr

    def solve() -> float:
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        solver.pcg.solve_mixed_block(solver.precond, b, x, 1e-30, 400, stream)
        return time.perf_counter() - t

    solve()
    stamps: list[float] = []
    running = threading.Event()
    done = threading.Event()

    def tick() -> None:
        running.set()
        while not done.is_set():
            stamps.append(time.perf_counter())

    worker = threading.Thread(target=tick)
    worker.start()
    running.wait()
    start = time.perf_counter()
    elapsed = solve()
    end = time.perf_counter()
    done.set()
    worker.join()
    inside = [t for t in stamps if start <= t <= end]
    gap = max(np.diff([start, *inside, end]))
    json.dump(
        {"gil": sys._is_gil_enabled(), "solve_s": elapsed, "max_gap_s": float(gap)}, sys.stdout
    )


if __name__ == "__main__":
    main()

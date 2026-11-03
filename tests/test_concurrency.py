"""Threads sharing the library, on whatever build this interpreter is.

Under the free-threaded build nothing else in the suite runs two threads at once, though the locks
in ``Subject``, the solver and the copier exist for exactly that. Results must match a serial run
bit for bit, whichever thread gets to a lazy build first.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from cunibs import ConductivityUQConfig, Placement, Subject
from cunibs._memory import release_scratch, scratch_pool

pytestmark = [pytest.mark.gpu, pytest.mark.realmesh]

THREADS = 6
ROUNDS = 5
RECOVERIES = ("raw", "harmonic", "spr_global", "spr_tissue")


def _in_threads(*calls):
    """Each call on its own thread, released together, with its result or its exception raised."""
    barrier = threading.Barrier(len(calls))

    def run(call):
        barrier.wait()
        return call()

    with ThreadPoolExecutor(len(calls)) as pool:
        return [f.result() for f in [pool.submit(run, call) for call in calls]]


def _fields(result):
    return result.magnE.tobytes(), result.E.tobytes(), result.iterations


@pytest.fixture(scope="module")
def placements(patch_mesh):
    centroids = patch_mesh.nodes_mm[patch_mesh.skin_tris].mean(axis=1)
    top = centroids[np.argsort(centroids[:, 2])[-400:]]
    picked = top[:: len(top) // 20][:20]
    return [
        Placement(c, c + 40.0 * np.array([np.cos(a), np.sin(a), 0.0]))
        for c, a in zip(picked, np.linspace(0, 6.0, len(picked)), strict=True)
    ]


def _sweep(subject, coil, placements, recovery="harmonic"):
    return [
        _fields(r)
        for r in subject.iter_simulate(
            coil, placements, 1e6, recovery=recovery, magnitude=True, vectors=True
        )
    ]


@pytest.fixture(scope="module")
def serial(patch_mesh, d70_coil, placements):
    with Subject(patch_mesh) as subject:
        return {r: _sweep(subject, d70_coil, placements, r) for r in RECOVERIES}


def test_threads_racing_a_cold_subject_match_serial(patch_mesh, d70_coil, placements, serial):
    for _ in range(ROUNDS):
        with Subject(patch_mesh) as subject:
            got = _in_threads(
                *[
                    lambda i=i: _fields(
                        subject.simulate(
                            d70_coil, placements[i], 1e6, magnitude=True, vectors=True
                        )
                    )
                    for i in range(THREADS)
                ]
            )
        assert got == serial["harmonic"][:THREADS]


def test_sweeps_sharing_a_subject_match_serial(patch_mesh, d70_coil, placements, serial):
    for _ in range(ROUNDS):
        with Subject(patch_mesh) as subject:
            modes = [RECOVERIES[i % len(RECOVERIES)] for i in range(THREADS)]
            got = _in_threads(
                *[lambda m=m: _sweep(subject, d70_coil, placements, m) for m in modes]
            )
        for mode, sweep in zip(modes, got, strict=True):
            assert sweep == serial[mode]


def test_sweeps_on_their_own_streams_match_serial(cp, patch_mesh, d70_coil, placements, serial):
    def on_a_stream(subject, mode):
        with cp.cuda.Stream(non_blocking=True):
            return _sweep(subject, d70_coil, placements, mode)

    for _ in range(ROUNDS):
        with Subject(patch_mesh) as subject:
            modes = [RECOVERIES[i % len(RECOVERIES)] for i in range(THREADS)]
            got = _in_threads(*[lambda m=m: on_a_stream(subject, m) for m in modes])
        for mode, sweep in zip(modes, got, strict=True):
            assert sweep == serial[mode]


def test_separate_subjects_match_serial(patch_mesh, d70_coil, placements, serial):
    def alone():
        with Subject(patch_mesh) as subject:
            return _sweep(subject, d70_coil, placements)

    for _ in range(ROUNDS):
        for sweep in _in_threads(*[alone] * THREADS):
            assert sweep == serial["harmonic"]


def test_uq_alongside_sweeps_matches_serial(patch_mesh, d70_coil, placements, serial):
    config = ConductivityUQConfig(n_samples=8, seed=0)

    def uq(subject):
        result = subject.simulate_conductivity_uq(
            d70_coil, placements[0], config, 1e6, moments=True
        )
        return result.mean_magnE.tobytes(), result.std_magnE.tobytes()

    with Subject(patch_mesh) as subject:
        want = uq(subject)
    for _ in range(ROUNDS):
        with Subject(patch_mesh) as subject:
            first, sweep, second = _in_threads(
                lambda s=subject: uq(s),
                lambda s=subject: _sweep(s, d70_coil, placements),
                lambda s=subject: uq(s),
            )
        assert first == want
        assert second == want
        assert sweep == serial["harmonic"]


def _sweeps_while_freeing(subject, coil, placements):
    stop = threading.Event()

    def keep_freeing():
        while not stop.is_set():
            subject.free()

    freer = threading.Thread(target=keep_freeing)
    freer.start()
    try:
        return _in_threads(*[lambda: _sweep(subject, coil, placements) for _ in range(THREADS)])
    finally:
        stop.set()
        freer.join()
        subject.free()


def test_free_during_sweeps_leaves_them_correct(patch_mesh, d70_coil, placements, serial):
    """free() may rebuild what a running sweep already holds, never change or break it."""
    for _ in range(ROUNDS):
        for sweep in _sweeps_while_freeing(Subject(patch_mesh), d70_coil, placements):
            assert sweep == serial["harmonic"]


def test_concurrent_sweeps_release_their_device_arrays(patch_mesh, d70_coil, placements):
    """With the workers idle. On a free-threaded build a release left to another thread would wait."""
    release_scratch()
    with ThreadPoolExecutor(THREADS) as pool, Subject(patch_mesh) as subject:
        futures = [pool.submit(_sweep, subject, d70_coil, placements) for _ in range(THREADS)]
        for future in futures:
            future.result()
        assert scratch_pool().used_bytes() == 0
    assert scratch_pool().total_bytes() == 0

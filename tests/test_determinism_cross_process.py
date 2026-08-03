"""The AMG hierarchy must come out identical in a fresh interpreter, not just a fresh call.

Every other determinism test in this suite runs inside one pytest process with warm caches,
which cannot see the failure mode this file exists for: cuSPARSE chooses how to split a row
from runtime state, so two calls in one process can agree bit for bit while two processes do
not. Without this, the README's "repeating a run reproduces it bitwise" is a per-process claim.

Marked ``reference`` deliberately. The patch mesh cannot answer the question: at 8403 rows
every stage is already bit-identical across processes, because cuSPARSE only repartitions once
rows get long.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

CHILD = Path(__file__).parent / "_hierarchy_child.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
_CHILD_ATTEMPTS = 3


def _spawn(args: list[str], env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the child, retrying only one killed by a signal.

    A child building a full-mesh hierarchy while another process in the session holds a CUDA
    context intermittently dies inside ``select_size4`` with an illegal access or a segfault.
    The fixture keeps this test's own parent out of CUDA, but a full-suite run has already put
    a context there. A child that raised and exited on its own has produced a real answer --
    an OutOfMemoryError when something else on the card is holding memory is the common one --
    and retrying it only delays reporting it under a message about crashing.
    """
    for _ in range(_CHILD_ATTEMPTS):
        proc = subprocess.run(  # noqa: S603  # this interpreter, a file in this directory
            [sys.executable, "-u", str(CHILD), *args],
            capture_output=True,
            text=True,
            env=os.environ | env_overrides,
            cwd=REPO_ROOT,
            timeout=900,
            check=False,
        )
        if proc.returncode >= 0:
            break
    return proc


def _run_child(operator_dir: Path, env_overrides: dict[str, str]) -> dict[str, str]:
    """One fresh interpreter; returns the stage name -> digest map it printed."""
    proc = _spawn([str(operator_dir)], env_overrides)
    if proc.returncode != 0:
        pytest.fail(f"child failed (rc={proc.returncode})\n{proc.stderr[-2000:]}")
    hashes = {}
    for line in proc.stdout.splitlines():
        if line.startswith("CHILD_HASH "):
            _, name, digest = line.split()
            hashes[name] = digest
        elif line.startswith("CHILD_SIZES "):
            hashes["sizes"] = line.split(maxsplit=1)[1]
    if not hashes:
        pytest.fail(f"child printed no hashes\n{proc.stdout[-2000:]}")
    return hashes


@pytest.fixture
def saved_operator(reference_mesh_path, tmp_path) -> Path:
    """The reduced fp32 operator, assembled once and handed to every child as bytes.

    Assembled in a subprocess, so this test never brings a CUDA context into the pytest
    process; see ``_spawn``.

    Handing the children an operator rather than a mesh also keeps the assembly out of what is
    being compared. A head mesh carries near-degenerate tetrahedra, and the batched solve behind
    ``gradient_operator`` takes a pivoting path for those that depends on how much device memory
    is free. That is a real and separate question; folding it in would only make this test flaky
    about something it is not asking.
    """
    path = tmp_path / "operator"
    proc = _spawn(["--save-operator", str(reference_mesh_path), str(path)], {})
    if proc.returncode != 0:
        pytest.fail(
            f"could not assemble the operator (rc={proc.returncode})\n{proc.stderr[-2000:]}"
        )
    return path


@pytest.mark.reference
@pytest.mark.slow
def test_hierarchy_is_reproducible_across_processes(saved_operator, tmp_path):
    """Three fresh interpreters, one with a cold kernel cache, must agree exactly.

    The cold-cache run is the interesting one: CuPy compiles kernels at runtime, so a cold
    ``CUPY_CACHE_DIR`` is what a new machine has, and it is the state a user re-running the
    analysis tomorrow is most likely to be in.

    ``PYTHON_GIL=0`` is deliberately not one of the axes here. Free-threaded NumPy has a
    separate, known sorting race that would fail this test for an unrelated reason;
    ``benchmarks/manuscript_determinism.py`` part D covers that axis where it can be reported
    rather than merely failing.
    """
    runs = [
        _run_child(saved_operator, {}),
        _run_child(saved_operator, {}),
        _run_child(saved_operator, {"CUPY_CACHE_DIR": str(tmp_path / "cold")}),
    ]

    reference, *others = runs
    assert "l0.dinv" in reference, f"child emitted no levels: {sorted(reference)}"
    assert reference["operator"] == runs[1]["operator"], "children read different operators"
    for i, other in enumerate(others, start=2):
        assert other.keys() == reference.keys(), f"run {i} produced different stages"
        moved = [name for name, digest in other.items() if digest != reference[name]]
        assert not moved, f"run {i} differs from run 1 at: {', '.join(sorted(moved))}"

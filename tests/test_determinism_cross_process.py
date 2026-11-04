"""A run must come out identical in a fresh interpreter, not just a fresh call.

Every other determinism test in this suite runs inside one pytest process with warm caches, which
cannot see what a process carries: cuSPARSE chooses how to split a row from runtime state, so two
calls in one process can agree bit for bit while two processes do not. Without this, the README's
"repeating a run reproduces it bitwise" is a per-process claim.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

CHILD = Path(__file__).parent / "_hierarchy_child.py"
REPEAT_CHILD = Path(__file__).parent / "_repeat_child.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _spawn(args: list[str], env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # this interpreter, a file in this directory
        [sys.executable, "-u", str(CHILD), *args],
        capture_output=True,
        text=True,
        env=os.environ | env_overrides,
        cwd=REPO_ROOT,
        timeout=900,
        check=False,
    )


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
    """The reduced fp32 operator, assembled once and handed to every child as bytes."""
    path = tmp_path / "operator"
    proc = _spawn(["--save-operator", str(reference_mesh_path), str(path)], {})
    if proc.returncode != 0:
        pytest.fail(
            f"could not assemble the operator (rc={proc.returncode})\n{proc.stderr[-2000:]}"
        )
    return path


@pytest.mark.realmesh
def test_a_fresh_interpreter_with_the_gil_reproduces_a_run(patch_mesh_path):
    """A cold process against this warm one, and on a free-threaded build, GIL on against off."""
    proc = subprocess.run(  # noqa: S603  # this interpreter, a file in this directory
        [sys.executable, str(REPEAT_CHILD), str(patch_mesh_path)],
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHON_GIL": "1"},
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    fresh = json.loads(proc.stdout.strip().splitlines()[-1])
    assert fresh == runpy.run_path(str(REPEAT_CHILD))["digests"](patch_mesh_path)


@pytest.mark.reference
@pytest.mark.slow
def test_hierarchy_is_reproducible_across_processes(saved_operator, tmp_path):
    """Three fresh interpreters, one with a cold kernel cache, must agree exactly.

    The cold-cache run is the interesting one: CuPy compiles kernels at runtime, so a cold
    ``CUPY_CACHE_DIR`` is what a new machine has, and it is the state a user re-running the
    analysis tomorrow is most likely to be in.

    Marked ``reference`` because at the patch's 8403 rows every stage is already bit-identical
    across processes: cuSPARSE only repartitions once rows get long.
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

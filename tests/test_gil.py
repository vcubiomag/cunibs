"""A long native call must let other Python threads run, including on builds with a GIL."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.realmesh]

CHILD = Path(__file__).parent / "_gil_child.py"


def test_a_block_solve_does_not_hold_the_gil(patch_mesh_path):
    proc = subprocess.run(  # noqa: S603  # this interpreter, a file in this directory
        [sys.executable, str(CHILD), str(patch_mesh_path)],
        env={**os.environ, "PYTHON_GIL": "1"},
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["gil"]
    assert result["max_gap_s"] < 0.25 * result["solve_s"], result

"""The benchmark helpers whose mistakes would only show up as a wrong published number."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
CHILD = REPO / "benchmarks" / "simnibs_child.py"


@pytest.fixture(autouse=True)
def _benchmarks_importable(monkeypatch):
    monkeypatch.syspath_prepend(REPO)


def test_failed_request_can_be_answered_again(tmp_path):
    request = tmp_path / "abc.request.npz"
    np.savez(request, __request__=np.frombuffer(json.dumps({"op": "nope"}).encode(), np.uint8))

    serve = [sys.executable, str(CHILD), "serve", str(tmp_path)]
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(serve, check=True, capture_output=True)  # noqa: S603


@pytest.mark.gpu
def test_free_pools_returns_the_scratch_pool():
    import cupy as cp

    from benchmarks import _measure
    from cunibs._memory import scratch_pool

    with cp.cuda.using_allocator(scratch_pool().malloc):
        cp.empty(1024)
    assert scratch_pool().free_bytes() > 0
    _measure.free_pools()
    assert scratch_pool().total_bytes() == 0


def test_field_metrics_on_values_worked_by_hand():
    from benchmarks import _fields

    nodes = np.array([[0.0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 6]])
    volumes = _fields.tet_volumes(nodes, np.array([[0, 1, 2, 3], [1, 0, 2, 3]]))
    assert volumes.tolist() == pytest.approx([6.0, 6.0])

    x, w = np.array([3.0, 1.0, 2.0], np.float32), np.array([1.0, 2.0, 1.0])
    assert [_fields.weighted_percentile(x, w, q) for q in (50, 51, 76)] == [1.0, 2.0, 3.0]

    assert _fields.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert _fields.spearman([1, 2, 2, 3], [4, 3, 3, 1]) == pytest.approx(-1.0)
    assert _fields.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(2 / 5**0.5)

    fit = _fields.fit_power([1.0, 2.0, 4.0], [3.0, 12.0, 48.0])
    assert (fit["slope"], fit["a"], fit["r2"]) == pytest.approx((2.0, 3.0, 1.0))
